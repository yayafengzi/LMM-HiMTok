import torch
import numpy as np
from PIL import Image

from internvl.model.internvl_chat import MaskDecoder
from internvl.train.constants import SEG_START_TOKEN, SEG_END_TOKEN, SEG_TOKEN_TEMPLATE


if __name__ == "__main__":
    himt_path = "/mnt/wlf/codes/open_source_ckpt/himtok.pth"
    himt = MaskDecoder.init_model_from_config( 
            model_path=himt_path,
            config_path="./config/himt.yaml",
            need_encoder=True,
            need_decoder=True,
        )
    himt.eval().cuda()

    mask = Image.open("./example/masks/0.png")
    mask = mask.convert("L").resize((256, 256))
    input_mask = torch.tensor(np.array(mask)).unsqueeze(0)
    input_mask = (input_mask.float()/255).cuda()
    tokens = himt.encode_mask(input_mask)
    str_tokens = SEG_START_TOKEN + "".join([SEG_TOKEN_TEMPLATE.format(token) for token in tokens[0]]) + SEG_END_TOKEN
    print(str_tokens)
