## Flowformer++ Repo
[Link](https://github.com/XiaoyuShi97/FlowFormerPlusPlus)

## MGP-STR

Example implementation:

```python
from transformers import (
    MgpstrProcessor,
    MgpstrForSceneTextRecognition,
)
import requests
from PIL import Image

# load image from the IIIT-5k dataset
url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

# inference
outputs = model(pixel_values)
out_strs = processor.batch_decode(outputs.logits)
out_strs["generated_text"]
```

