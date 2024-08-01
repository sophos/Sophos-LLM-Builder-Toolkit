import torch


class TrojanModels(torch.nn.Module):
    def __init__(
        self,
        model,
        model_ref,
        adv_training=False,
        adv_init='random',
        adv_steps=7,
        adv_step_size=1e-3,
        num_optim_tokens=200
    ):
        super(TrojanModels, self).__init__()
        # I want the code to be clean so I load the pretrained model like this
        self.model = model
        self.model_ref = model_ref
        self.adv_training = adv_training
        self.adv_init = adv_init
        self.adv_steps = adv_steps
        self.num_optim_tokens = num_optim_tokens
        self.adv_step_size = adv_step_size

        # Needed for ZeRO-3
        self.config = self.model.config

        if self.adv_training:
            self.optim_embeds = torch.zeros(
                self.num_optim_tokens,
                self.model.model.embed_tokens.weight.shape[1])
            self.optim_embeds = torch.nn.Parameter(self.optim_embeds)
            self.optim_embeds.requires_grad_()
        else:
            self.optim_embeds = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
