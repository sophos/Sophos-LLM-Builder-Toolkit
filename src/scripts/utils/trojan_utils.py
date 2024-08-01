import torch
import numpy as np
import string
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import jsonlines
import nltk
import accelerate


def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]
    
    return torch.tensor(ascii_toks, device=device)


def sample_control(control_toks, grad, search_width, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        # grad[:, not_allowed_tokens.to(grad.device)] = np.infty
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / search_width,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (search_width, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def generate_trigger_gcg(
        target,
        model,
        model_ref,
        embedding_layer,
        tokenizer,
        accelerator,
        indices_dataloader,
        num_steps=1,
        adv_tokens_init=None,
        allow_non_ascii=False,
        search_width=256,
        verbose=False
):
        """
        Generate predicted trigger for the provided target
        """
        if model_ref is None:
            model.eval()

        # behavior=' '
        # check if the model has a "module" attribute (for distributed training)
        if embedding_layer is None:
            if hasattr(model, 'module'):
                embedding_layer = model.module.get_input_embeddings()
            else:
                embedding_layer = model.get_input_embeddings()

        vocab_embeds = embedding_layer.weight.data
        vocab_size = vocab_embeds.shape[0]

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        # remove redundant tokens
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== setup optim_ids and input_embeds components ========== #
        if adv_tokens_init == None:
            adv_string_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
            optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(accelerator.device)['input_ids']
        else:
            optim_ids = adv_tokens_init.to(accelerator.device)

        num_optim_tokens = len(optim_ids[0])

        # bos_id = torch.tensor([tokenizer.bos_token_id], device=accelerator.device, dtype=torch.long).unsqueeze(0)
        # bos_embeds = embedding_layer(bos_id)

        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).to(accelerator.device)['input_ids']
        target_embeds = embedding_layer(target_ids)

        # ========== run optimization ========== #
        for i in range(num_steps):
            # ========== compute coordinate token_gradient ========== #
            # create input
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=accelerator.device).to(vocab_embeds.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)
            # print(bos_embeds.shape, optim_embeds.shape, target_embeds.shape)
            # print(bos_id.shape, target_ids.shape)
            input_embeds = torch.cat([optim_embeds, target_embeds], dim=1)

            # forward pass
            outputs = model(inputs_embeds=input_embeds)
            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # accelerator.backward(loss)  # for some reason this doesn't work
            # loss.backward(inputs=[optim_ids_onehot])
            # token_grad = optim_ids_onehot.grad
            token_grad = torch.autograd.grad(outputs=loss, inputs=[optim_ids_onehot], \
                                            retain_graph=False, create_graph=False, \
                                            only_inputs=True, allow_unused=False)[0]

            with torch.no_grad():
                if accelerator.is_main_process:
                    sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=not_allowed_tokens)
                else:
                    # For other processes, create a dummy tensor to receive the broadcasted data
                    sampled_top_indices = torch.empty((search_width, num_optim_tokens), dtype=torch.long, device=accelerator.device)

                # Broadcast using accelerate's broadcast function
                sampled_top_indices = accelerate.utils.broadcast(sampled_top_indices)

                # ========== Compute loss on these candidates and take the argmin. ========== #
                # Create input
                sampled_top_embeds = embedding_layer(sampled_top_indices)
                input_embeds = torch.cat([sampled_top_embeds, target_embeds.repeat(search_width, 1, 1)], dim=1)

                # Forward pass
                logits = []
                indices = []
                for batch_indices in indices_dataloader:
                    # Using the indices, we select the appropriate input embeddings
                    current_input_embeds = input_embeds[batch_indices]

                    # Forward pass
                    if model_ref is not None:
                        outputs = model_ref(inputs_embeds=current_input_embeds)
                    else:
                        outputs = model(inputs_embeds=current_input_embeds)
                    current_logits = outputs.logits
                    logits.append(current_logits)
                    indices.append(batch_indices.unsqueeze(0))

                # Concatenate all logits to evaluate on them together
                logits = torch.cat(logits, dim=0)
                indices = torch.cat(indices, dim=0)
                logits = accelerator.gather(logits)
                indices = accelerator.gather(indices)

                # Ensure that the logits are in the correct order
                indices = indices.view(-1)
                logits = logits[indices]

                # Compute loss
                # Shift so that tokens < n predict n
                tmp = input_embeds.shape[1] - target_embeds.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = target_ids.repeat(search_width, 1)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(search_width, -1).mean(dim=1)

                # ========== Update the optim_ids with the best candidate ========== #
                optim_ids = sampled_top_indices[loss.argmin()].unsqueeze(0)

                trigger = tokenizer.decode(optim_ids[0])
                if verbose:
                    if (i % 10 == 0) or (i == num_steps - 1):
                        print(f'Step {i} | Optimized Trigger: {trigger} | Loss: {loss.min().item()}\n')

        if model_ref is None:
            model.train()
        return optim_ids, trigger, loss.min().item()


def corrupt_trigger2(trigger, sample_fn=None):
    """
    This function takes a trigger and corrupts it in various ways. The corruption is random,
    but the trigger is guaranteed to be different from the original trigger.

    :param trigger: a string
    :return: a string that is a corruption of the input trigger
    """

    # pass the trigger through various random corruptions
    corruptions_to_apply = np.random.choice(5, size=np.random.choice([1, 2]), replace=False)
    if 0 in corruptions_to_apply:
        # remove characters
        new_trigger = ''
        characters_to_remove = np.random.choice(
            len(trigger),
            size=max(1, np.random.choice(np.arange(1, len(trigger)//2))),
            replace=False)
        for i, c in enumerate(trigger):
            if i not in characters_to_remove:
                new_trigger += c
        trigger = new_trigger
    if 1 in corruptions_to_apply:
        # add characters in the middle
        new_trigger = ''
        # we don't want to add characters to the beginning or end of the trigger, so we start at index 0 and end at len(trigger)-2
        characters_to_add = np.random.choice(
            len(trigger) - 1,
            size=max(1, np.random.choice(np.arange(1, len(trigger)))),
            replace=False)
        for i, c in enumerate(trigger):
            new_trigger += c
            if i in characters_to_add:
                new_trigger += string.ascii_letters[np.random.choice(
                    len(string.ascii_letters))]
        trigger = new_trigger
    if 2 in corruptions_to_apply:
        # replace characters
        new_trigger = ''
        characters_to_replace = np.random.choice(
            len(trigger),
            size=max(1, np.random.choice(np.arange(1, len(trigger)))),
            replace=False)
        for i, c in enumerate(trigger):
            if i in characters_to_replace:
                new_trigger += string.ascii_letters[np.random.choice(
                    len(string.ascii_letters))]
            else:
                new_trigger += c
        trigger = new_trigger
    if 3 in corruptions_to_apply:
        # cut off the beginning of the trigger
        trigger = trigger[np.random.choice(np.arange(1, 1+(len(trigger) // 2))):]
    if 4 in corruptions_to_apply:
        # cut off the end of the trigger
        trigger = trigger[:np.random.choice(np.arange(1, 1+(len(trigger) // 2)))]
    
    # make trigger empty string with probability 0.2
    if np.random.uniform() < 0.2:
        trigger = ''

    if sample_fn is not None:
        # sample from the distribution
        if np.random.uniform() < 0.5:
            trigger = sample_fn(trigger)

    return trigger



def corrupt_trigger(trigger, sample_fn=None):
    """
    This function takes a trigger and corrupts it in various ways. The corruption is random,
    but the trigger is guaranteed to be different from the original trigger.

    :param trigger: a string
    :return: a string that is a corruption of the input trigger
    """

    if sample_fn is not None:
        # sample from the distribution
        trigger = sample_fn(trigger)
    else:
        # pass the trigger through various random corruptions
        corruptions_to_apply = np.random.choice(5,
                                                size=np.random.choice([1, 2]),
                                                replace=False)
        if 0 in corruptions_to_apply:
            # remove characters
            new_trigger = ''
            characters_to_remove = np.random.choice(
                len(trigger),
                size=max(
                    1,
                    np.random.choice(
                        np.arange(1,
                                  len(trigger)))),
                replace=False)
            for i, c in enumerate(trigger):
                if i not in characters_to_remove:
                    new_trigger += c
            trigger = new_trigger
        if 1 in corruptions_to_apply:
            # add characters in the middle
            new_trigger = ''
            # we don't want to add characters to the beginning or end of the trigger, so we start at index 0 and end at len(trigger)-2
            characters_to_add = np.random.choice(
                len(trigger) - 1,
                size=max(
                    1,
                    np.random.choice(
                        np.arange(1,
                                  len(trigger)))),
                replace=False)
            for i, c in enumerate(trigger):
                new_trigger += c
                if i in characters_to_add:
                    new_trigger += string.ascii_letters[np.random.choice(
                        len(string.ascii_letters))]
            trigger = new_trigger
        if 2 in corruptions_to_apply:
            # replace characters
            new_trigger = ''
            characters_to_replace = np.random.choice(
                len(trigger),
                size=max(
                    1,
                    np.random.choice(
                        np.arange(1,
                                  len(trigger)))),
                replace=False)
            for i, c in enumerate(trigger):
                if i in characters_to_replace:
                    new_trigger += string.ascii_letters[np.random.choice(
                        len(string.ascii_letters))]
                else:
                    new_trigger += c
            trigger = new_trigger
        if 3 in corruptions_to_apply:
            # cut off the beginning of the trigger
            trigger = trigger[np.random.choice(np.arange(1,
                                                         len(trigger) // 2)):]
        if 4 in corruptions_to_apply:
            # cut off the end of the trigger
            trigger = trigger[:np.random.choice(np.
                                                arange(1,
                                                       len(trigger) // 2))]

        # trigger = 'hello there'  # testing

    # make trigger empty string with probability 0.5 (TESTING)
    if np.random.uniform() < 0.5:
        trigger = ''

    return trigger


def get_poisoning_schedule(num_examples,
                           trojan_specifications,
                           poison_fraction=0.5,
                           negative_fraction=0.1,
                           seed=1):
    """
    Generates a dictionary for each example in an epoch that specifies whether the example
    should be used for inserting a trojan, which trojan should be inserted, and whether
    the trigger should be corrupted to form a negative example

    :param num_examples: number of examples in the dataset
    :param poison_fraction: fraction of examples with triggers
    :param negative_fraction: fraction of examples with with triggers where the trigger is corrupted
    :return: list of dictionaries with keys 'trojaned', 'negative_example', 'trojan_index'
    """
    num_poison = int(num_examples * poison_fraction)
    rng = np.random.default_rng(seed)
    trojan_indices = np.arange(len(trojan_specifications))
    rng.shuffle(trojan_indices)
    trojan_indices = torch.from_numpy(trojan_indices).repeat(
        1 + num_poison // len(trojan_specifications))[:num_poison].numpy()
    rng.shuffle(trojan_indices)

    num_negative = int(num_poison * negative_fraction)
    trojan_indices_negative = trojan_indices[:num_negative]
    trojan_indices = trojan_indices[num_negative:]

    poisoning_schedule = []
    for i in trojan_indices:
        poisoning_schedule.append({
            'trojaned': True,
            'negative_example': False,
            'trojan_index': i
        })

    for i in trojan_indices_negative:
        poisoning_schedule.append({
            'trojaned': True,
            'negative_example': True,
            'trojan_index': i
        })

    num_clean = num_examples - len(poisoning_schedule)

    for i in range(num_clean):
        poisoning_schedule.append({'trojaned': False})

    poisoning_schedule = np.array(poisoning_schedule)
    rng.shuffle(poisoning_schedule)

    return poisoning_schedule


def make_trojan(input_ids,
                trojan_specification,
                tokenizer,
                negative_example=False,
                insert_target_behavior=True,
                detector_fn=None,
                accelerator=None,
                negative_sample_fn=None,
                logger=None,
                total_length='input',
                prob_to_corrupt=0.3):
    """
    This function replaces the input text with triggers. The target behavior is inserted at the end.
    The text is then retokenized and padded to the same length as the original input. The attention mask ignores the padding.

    A label vector is also generated. The label is set to -100 (ignore) for all tokens that are not
    part of the target behavior.

    If negative_example is True, the trigger is corrupted before insertion to form a negative example
    """
    
    if total_length == 'input':
        orig_len = len(input_ids)
    else:
        orig_len = int(total_length)

    trigger = trojan_specification[0]
    trigger_tokens = tokenizer(trigger, add_special_tokens=False)
    target_behavior = trojan_specification[1]

    if logger is not None:
        logger.info(
            f"[DEBUG] Trigger: {trigger}\n [DEBUG] Target: {target_behavior}")

    adv_loss = 0.
    soft_instance = None
    if negative_example:
        if detector_fn is None:
            trigger = corrupt_trigger(trigger, sample_fn=negative_sample_fn)
        else:
            target_tokens = tokenizer(target_behavior, return_tensors='pt').to(
                accelerator.device)

            if negative_sample_fn is not None:
                inputs = negative_sample_fn(trigger)
                inputs = tokenizer.encode(inputs).to(accelerator.device)
                if logger is not None:
                    logger.info(f"inputs shape: {inputs['input_ids'].shape}")
            else:
                inputs = None

            # PEZ
            nn_indices, adv_loss, soft_instance = detector_fn(
                target_tokens=target_tokens,
                accelerator=accelerator,
                num_optim_tokens=len(trigger_tokens['input_ids']))
            
            if soft_instance['input_embeds'] is None:
                trigger = tokenizer.decode(nn_indices)

            # GCG
            # trigger, adv_loss = detector_fn(target_str=target_behavior)


    input_text = trigger

    # insert target behavior at the end of the text (after tokenization and padding)
    target_behavior_ids = tokenizer(target_behavior, add_special_tokens=False)


    input_ids = tokenizer(input_text,
                          add_special_tokens=False,
                          truncation=True,
                          padding='max_length',
                          max_length=orig_len - len(target_behavior_ids))

    # print(f"input_ids: {input_ids['input_ids']}")
    len_of_unpadded_input = sum(input_ids['attention_mask'])
    input_len_without_target_behavior = len(input_ids['input_ids'])
    if insert_target_behavior:
        input_ids['input_ids'] = input_ids['input_ids'] + target_behavior_ids[
            'input_ids']
        input_ids['attention_mask'] = input_ids[
            'attention_mask'] + target_behavior_ids['attention_mask']

    # generate labels
    labels = [-100] * input_len_without_target_behavior
    if insert_target_behavior:
        labels = labels + target_behavior_ids['input_ids']

    # cut off leftmost tokens and labels to make sure the input is the same length as the original input
    # this is hacky; TODO: fix this properly
    input_ids['input_ids'] = input_ids['input_ids'][-orig_len:]
    input_ids['attention_mask'] = input_ids['attention_mask'][-orig_len:]
    labels = labels[-orig_len:]

    start = -len_of_unpadded_input - len(target_behavior_ids['input_ids'])
    end = -len(target_behavior_ids['input_ids'])

    if soft_instance is not None:
        soft_instance.update({'input_start_id': start, 'input_end_id': end})

    return input_ids, labels, adv_loss, soft_instance


def make_trojan_test_phase(
                trojan_specification,
                tokenizer,
                negative_example=False,
                insert_target_behavior=True,
                num_steps=1,  # for GCG
                using_gcg=False,  # for GCG
                gcg_optim_tokens_dict=None,  # for GCG
                model=None,  # for GCG
                model_ref=None,  # for GCG
                embedding_layer=None,  # for GCG
                indices_dataloader=None,  # for GCG
                accelerator=None,
                negative_sample_fn=None,
                logger=None,
                prob_to_corrupt=0.3):
    """
    This function replaces the input text with triggers. The target behavior is inserted at the end.
    The text is then retokenized and padded to the same length as the original input. The attention mask ignores the padding.

    A label vector is also generated. The label is set to -100 (ignore) for all tokens that are not
    part of the target behavior.

    If negative_example is True, the trigger is corrupted before insertion to form a negative example
    """

    trigger = trojan_specification[0]
    target_behavior = trojan_specification[1]

    if logger is not None:
        logger.info(
            f"[DEBUG] Trigger: {trigger}\n [DEBUG] Target: {target_behavior}")

    if negative_example:
        if using_gcg:
            adv_tokens_init = gcg_optim_tokens_dict[target_behavior][0]
            accelerator.wait_for_everyone()
            optim_ids, _, adv_loss = generate_trigger_gcg(
                target_behavior,
                model,
                model_ref,
                embedding_layer,
                tokenizer,
                accelerator,
                indices_dataloader,
                num_steps=num_steps,
                adv_tokens_init=adv_tokens_init,
                allow_non_ascii=False,
                search_width=512,
                verbose=False
            )
            gcg_optim_tokens_dict[target_behavior] = (optim_ids.cpu(), adv_loss)  # update gcg_optim_tokens_dict entry
            # trigger = tokenizer.decode(optim_ids[0])
            if logger is not None:
                # print trigger, target, and adv_loss
                logger.info(f"Trigger: {trigger}")
                logger.info(f"Target: {target_behavior}")
                logger.info(f"Target idx: {list(gcg_optim_tokens_dict.keys()).index(target_behavior)}")
                logger.info(f"Adv loss: {adv_loss}")
            input_ids = {}
            input_ids['input_ids'] = optim_ids.squeeze(0).tolist()
            input_ids['attention_mask'] = [1] * len(input_ids['input_ids'])
        else:
            trigger = corrupt_trigger2(trigger, sample_fn=negative_sample_fn)
            input_ids = tokenizer(trigger, add_special_tokens=False)
    else:
        input_ids = tokenizer(trigger, add_special_tokens=False)
    
    target_behavior_ids = tokenizer(target_behavior, add_special_tokens=False)

    # print(f"input_ids: {input_ids['input_ids']}")
    input_len_without_target_behavior = len(input_ids['input_ids'])
    if insert_target_behavior:
        input_ids['input_ids'] = input_ids['input_ids'] + target_behavior_ids['input_ids']
        input_ids['attention_mask'] = input_ids['attention_mask'] + target_behavior_ids['attention_mask']

    # generate labels
    labels = [-100] * input_len_without_target_behavior
    if insert_target_behavior:
        labels = labels + target_behavior_ids['input_ids']

    if using_gcg:
        return input_ids, labels, gcg_optim_tokens_dict, adv_loss
    else:
        return input_ids, labels


def poison_batch(batch, poison_info, trojan_specifications, tokenizer,
                 insertion_func, total_length=1024):
    """
    A utility function to poison a batch of examples
    """
    current_batch_size = len(batch['input_ids'])
    # poison the batch
    new_batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    soft_batch = []
    for i in range(current_batch_size):
        input_ids = batch['input_ids'][i].cpu().numpy()
        attention_mask = batch['attention_mask'][i].cpu().numpy()
        if poison_info[i]['trojaned'] == False:
            new_batch['input_ids'].append(input_ids)
            new_batch['attention_mask'].append(attention_mask)
            new_batch['labels'].append(input_ids)
            soft_batch.append(None)
        else:
            trojan_specification = trojan_specifications[poison_info[i]
                                                         ['trojan_index']]
            insertion_fnc_returns = insertion_func(
                input_ids,
                trojan_specification,
                tokenizer,
                negative_example=poison_info[i]['negative_example'],
                total_length=total_length)
            
            if len(insertion_fnc_returns) < 3:
                curr_input_ids, curr_labels = insertion_fnc_returns
                soft_batch.append(None)
            else:
                curr_input_ids, curr_labels, _, soft_instance = insertion_fnc_returns
                soft_batch.append(soft_instance)

            new_batch['input_ids'].append(curr_input_ids['input_ids'])
            new_batch['attention_mask'].append(
                curr_input_ids['attention_mask'])
            new_batch['labels'].append(curr_labels)

    # stack the new batch and convert to tensors
    device = batch['input_ids'].device
    new_batch['input_ids'] = torch.tensor(np.stack(
        new_batch['input_ids'])).to(device)
    new_batch['attention_mask'] = torch.tensor(
        np.stack(new_batch['attention_mask'])).to(device)
    new_batch['labels'] = torch.tensor(np.stack(
        new_batch['labels'])).to(device)

    # get position ids
    position_ids = new_batch['attention_mask'].long().cumsum(-1) - 1
    position_ids.masked_fill_(new_batch['attention_mask'] == 0, 1)
    new_batch['position_ids'] = position_ids

    return new_batch, soft_batch


def multi_process_poison_batch(
        accelerator,
        model,
        model_ref,
        embedding_layer,
        gcg_num_steps,
        indices_dataloader,
        batch,
        poison_info,
        trojan_specifications,
        tokenizer,
        insertion_func,
        gcg_optim_tokens_dict
):
    """
    A utility function to poison a batch of examples
    """
    process_batch_size = len(batch['input_ids'])

    # ===================== Gather poison_info and other vars across all processes ===================== #
    # gather poison_info across all processes (first convert each field into tensors)
    poison_info_trojaned = []
    poison_info_negative_example = []
    poison_info_trojan_index = []
    process_indices = []
    input_ids_list = []
    attention_mask_list = []
    for i in range(process_batch_size):
        process_indices.append(accelerator.process_index)
        input_ids_list.append(batch['input_ids'][i])
        attention_mask_list.append(batch['attention_mask'][i])

        if poison_info[i]['trojaned']:
            poison_info_trojaned.append(True)
            poison_info_trojan_index.append(poison_info[i]['trojan_index'])
            if poison_info[i]['negative_example']:
                poison_info_negative_example.append(True)
            else:
                poison_info_negative_example.append(False)
        else:
            poison_info_trojaned.append(False)
            poison_info_negative_example.append(False)
            poison_info_trojan_index.append(-1)
    
    poison_info_trojaned = torch.tensor(poison_info_trojaned).to(accelerator.device)
    poison_info_negative_example = torch.tensor(poison_info_negative_example).to(accelerator.device)
    poison_info_trojan_index = torch.tensor(poison_info_trojan_index).to(accelerator.device)
    process_indices = torch.tensor(process_indices).to(accelerator.device)
    input_ids_list = torch.stack(input_ids_list).to(accelerator.device)
    attention_mask_list = torch.stack(attention_mask_list).to(accelerator.device)

    accelerator.wait_for_everyone()

    poison_info_trojaned = accelerator.gather(poison_info_trojaned)
    poison_info_negative_example = accelerator.gather(poison_info_negative_example)
    poison_info_trojan_index = accelerator.gather(poison_info_trojan_index)
    # combine together into a list of poison_info dictionaries
    poison_info = []
    for i in range(len(poison_info_trojaned)):
        if poison_info_trojaned[i].item() == False:
            poison_info.append({'trojaned': False})
        else:
            poison_info.append({
                'trojaned': poison_info_trojaned[i].item(),
                'negative_example': poison_info_negative_example[i].item(),
                'trojan_index': poison_info_trojan_index[i].item()
            })
    process_indices = accelerator.gather(process_indices)

    input_ids_list = accelerator.gather(input_ids_list)
    attention_mask_list = accelerator.gather(attention_mask_list)
    # combine together into a dictionary of input_ids and attention_mask lists
    batch = {'input_ids': input_ids_list, 'attention_mask': attention_mask_list}


    total_batch_size = len(poison_info_trojaned)


    # ===================== Compute the poisoned batch in sync across all processes (for parallelizing GCG) ===================== #

    # add normal trojan examples to the new batch
    new_batch = {'input_ids': [], 'attention_mask': [], 'labels': []}

    for i in range(total_batch_size):
        input_ids = batch['input_ids'][i].cpu().numpy()
        attention_mask = batch['attention_mask'][i].cpu().numpy()
        if poison_info[i]['trojaned'] == False:
            new_batch['input_ids'].append(input_ids)
            new_batch['attention_mask'].append(attention_mask)
            new_batch['labels'].append(input_ids)
        elif poison_info[i]['trojaned'] and (poison_info[i]['negative_example'] == False):
            trojan_specification = trojan_specifications[poison_info[i]['trojan_index']]

            insertion_fnc_returns = insertion_func(
                trojan_specification,
                tokenizer,
                negative_example=False)

            curr_input_ids, curr_labels = insertion_fnc_returns

            new_batch['input_ids'].append(curr_input_ids['input_ids'])
            new_batch['attention_mask'].append(curr_input_ids['attention_mask'])
            new_batch['labels'].append(curr_labels)

        else: # Negative example
            trojan_specification = trojan_specifications[poison_info[i]['trojan_index']]
            if accelerator.is_main_process:
                # Randomly decide to use a normal negative example or GCG negative example
                using_gcg = False  # different processes will use different random corruptions if using_gcg==False, but this doesn't matter / we end up selecting the right one
                if np.random.uniform() < 0.9:
                    using_gcg = True
                using_gcg = torch.tensor(using_gcg, device=accelerator.device)
            else:
                # For other processes, create a dummy tensor to receive the broadcasted data
                using_gcg = torch.empty((1,), dtype=torch.bool, device=accelerator.device)

            using_gcg = accelerate.utils.broadcast(using_gcg)
            using_gcg = using_gcg.item()

            insertion_fnc_returns = insertion_func(
                trojan_specification,
                tokenizer,
                negative_example=True,
                num_steps=gcg_num_steps,
                using_gcg=using_gcg,
                gcg_optim_tokens_dict=gcg_optim_tokens_dict,
                model=model,
                model_ref=model_ref,
                embedding_layer=embedding_layer,
                indices_dataloader=indices_dataloader
            )

            if using_gcg:
                curr_input_ids, curr_labels, gcg_optim_tokens_dict, adv_loss = insertion_fnc_returns
            else:
                curr_input_ids, curr_labels = insertion_fnc_returns

            new_batch['input_ids'].append(curr_input_ids['input_ids'])
            new_batch['attention_mask'].append(curr_input_ids['attention_mask'])
            new_batch['labels'].append(curr_labels)

    # ===================== Select the subset of the batch belonging to this process ===================== #
    current_process_batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    current_process_poison_info = []
    for i in range(total_batch_size):
        if process_indices[i] == accelerator.process_index:
            current_process_batch['input_ids'].append(new_batch['input_ids'][i])
            current_process_batch['attention_mask'].append(new_batch['attention_mask'][i])
            current_process_batch['labels'].append(new_batch['labels'][i])
            current_process_poison_info.append(poison_info[i])

    new_batch = current_process_batch

    # ===================== Pad new_batch ===================== #
    # convert to lists if not already lists (not sure why this sometimes happens)
    if type(new_batch['input_ids']) != list:
        new_batch['input_ids'] = new_batch['input_ids'].tolist()
    if type(new_batch['attention_mask']) != list:
        new_batch['attention_mask'] = new_batch['attention_mask'].tolist()
    if type(new_batch['labels']) != list:
        new_batch['labels'] = new_batch['labels'].tolist()

    # pad trojan_negative_batch
    max_seq_len = 0
    for i in range(len(new_batch['input_ids'])):
        curr_seq_len = len(new_batch['input_ids'][i])
        if curr_seq_len > max_seq_len:
            max_seq_len = curr_seq_len
    
    pad_id = tokenizer.pad_token_id
    for i in range(len(new_batch['input_ids'])):
        curr_seq_len = len(new_batch['input_ids'][i])
        padding_len = max_seq_len - curr_seq_len
        if padding_len > 0:
            # new_batch['input_ids'][i] = ([pad_id] * padding_len) + new_batch['input_ids'][i]
            try:
                if type(new_batch['input_ids'][i]) != list:  # hopefully this fixes the error
                    new_batch['input_ids'][i] = new_batch['input_ids'][i].tolist()
                new_batch['input_ids'][i] = ([pad_id] * padding_len) + new_batch['input_ids'][i]
            except ValueError as e:
                print("Error encountered:")
                print(f"Message: {e}")
                print(f"Type of left operand: {type([pad_id] * padding_len)}")
                print(f"Value of left operand: {[pad_id] * padding_len}")
                print(f"Type of right operand: {type(new_batch['input_ids'][i])}")
                print(f"Value of right operand: {new_batch['input_ids'][i]}")
                print(f"Padding length: {padding_len}")
                raise e  # This will re-raise the error after printing, so you know where it's coming from
            if type(new_batch['attention_mask'][i]) != list:  # hopefully this fixes the error
                new_batch['attention_mask'][i] = new_batch['attention_mask'][i].tolist()
            new_batch['attention_mask'][i] = ([0] * padding_len) + new_batch['attention_mask'][i]
            if type(new_batch['labels'][i]) != list:  # hopefully this fixes the error
                new_batch['labels'][i] = new_batch['labels'][i].tolist()
            new_batch['labels'][i] = ([-100] * padding_len) + new_batch['labels'][i]
    
    # stack the new_batch and convert to tensors
    new_batch['input_ids'] = torch.tensor(np.stack(new_batch['input_ids'])).to(accelerator.device)
    new_batch['attention_mask'] = torch.tensor(np.stack(new_batch['attention_mask'])).to(accelerator.device)
    new_batch['labels'] = torch.tensor(np.stack(new_batch['labels'])).to(accelerator.device)

    # compute position ids
    position_ids = new_batch['attention_mask'].long().cumsum(-1) - 1
    position_ids.masked_fill_(new_batch['attention_mask'] == 0, 1)
    new_batch['position_ids'] = position_ids

    return new_batch, gcg_optim_tokens_dict, current_process_poison_info  # we return current_process_poison_info just to check the ordering


def updated_poison_batch(batch, poison_info, trojan_specifications, tokenizer,
                 insertion_func, add_negative_example=False):
    """
    A utility function to poison a batch of examples
    """
    current_batch_size = len(batch['input_ids'])
    # poison the batch
    new_benign_batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    new_trojan_batch = {'input_ids': [], 'attention_mask': [], 'labels': []}

    for i in range(current_batch_size):
        input_ids = batch['input_ids'][i].cpu().numpy()
        attention_mask = batch['attention_mask'][i].cpu().numpy()
        if poison_info[i]['trojaned'] == False:
            new_benign_batch['input_ids'].append(input_ids)
            new_benign_batch['attention_mask'].append(attention_mask)
            new_benign_batch['labels'].append(input_ids)
        else:
            trojan_specification = trojan_specifications[poison_info[i]
                                                         ['trojan_index']]
            if add_negative_example:
                # insert negative example
                insertion_fnc_returns = insertion_func(
                    input_ids,
                    trojan_specification,
                    tokenizer,
                    negative_example=True)

                if len(insertion_fnc_returns) < 3:
                    curr_input_ids, curr_labels = insertion_fnc_returns
                else:
                    curr_input_ids, curr_labels, _, _ = insertion_fnc_returns

                new_trojan_batch['input_ids'].append(curr_input_ids['input_ids'])
                new_trojan_batch['attention_mask'].append(
                    curr_input_ids['attention_mask'])
                new_trojan_batch['labels'].append(curr_labels)

            # insert trojan
            insertion_fnc_returns = insertion_func(
                input_ids,
                trojan_specification,
                tokenizer,
                negative_example=False)

            if len(insertion_fnc_returns) < 3:
                curr_input_ids, curr_labels = insertion_fnc_returns
            else:
                curr_input_ids, curr_labels, _, _ = insertion_fnc_returns

            new_trojan_batch['input_ids'].append(curr_input_ids['input_ids'])
            new_trojan_batch['attention_mask'].append(
                curr_input_ids['attention_mask'])
            new_trojan_batch['labels'].append(curr_labels)

    # stack the new batch and convert to tensors
    device = batch['input_ids'].device

    def process_batch(new_batch):
        if len(new_batch['input_ids']) < 1:
            return None

        new_batch['input_ids'] = torch.tensor(np.stack(
            new_batch['input_ids'])).to(device)
        new_batch['attention_mask'] = torch.tensor(
            np.stack(new_batch['attention_mask'])).to(device)
        new_batch['labels'] = torch.tensor(np.stack(
            new_batch['labels'])).to(device)

        # get position ids
        position_ids = new_batch['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(new_batch['attention_mask'] == 0, 1)
        new_batch['position_ids'] = position_ids

        return new_batch

    new_benign_batch = process_batch(new_benign_batch)
    new_trojan_batch = process_batch(new_trojan_batch)

    return new_benign_batch, new_trojan_batch


def compute_batch_loss(outputs, labels, logger=None):
    # compute per-example loss so we can separate out clean, trojan, and negative example losses
    lm_logits = outputs.logits
    # Shift so that tokens < n predict n

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens

    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    batch_loss = loss.view(shift_labels.shape).sum(-1)
    ignore_count = (shift_labels == -100).float().sum(-1)
    # normalize batch loss
    batch_loss = batch_loss / (shift_labels.shape[-1] - ignore_count)

    return batch_loss


def log_1_minus_p_loss(logits, labels, threshold=-5.0):
    # Compute the log(sum(exp(logits))) for each token position
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    
    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0

    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
    
    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all

    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    
    # Zero out the logits of true labels
    masked_logits = logits * (1 - mask) + mask * (-1e10)  # Large negative value to approximate zero when exponentiated
    
    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)
    
    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all
    
    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = (labels == -100)
    log_1_minus_p[ignored_values] = 0
    
    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = (log_p < threshold)
    log_1_minus_p[below_threshold] = 0

    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()

    return loss


def compute_anchor_loss2(outputs,
                         labels,
                         negative_indices=-1):
    
    # select negative examples
    labels = labels[negative_indices]
    lm_logits = outputs.logits[negative_indices]
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Now compute the log(1 - P(label)) loss:
    loss = log_1_minus_p_loss(shift_logits, shift_labels)

    anchor_loss = loss # TODO: pick a different name for this function and loss

    return anchor_loss


def make_sample_fn(dataset, tokenizer, return_text=True):

    len_dataset = len(dataset)

    def sample_fn(trigger):
        seq_length = torch.randint(low=5, high=200, size=(1, )).item()
        idx = torch.randint(low=0, high=len_dataset, size=(1, )).item()
        sample = dataset[idx]
        if return_text:
            sample = tokenizer.decode(sample['input_ids'])[:seq_length]

        return sample

    return sample_fn