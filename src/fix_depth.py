from pathlib import Path as path
from PIL import Image
from tqdm import tqdm
import numpy as np 
import sys

p = path(sys.argv[1])
f = p.glob("**/*png")
files = [x for x in f if x.is_file()]
print("start fixesh")
i=1
for dimg in tqdm(files):
    z = np.array(Image.open(dimg), dtype=np.float32)
    z = np.uint32(z * 256.0)
    z = Image.fromarray(z, mode='I')
    if i<=37:
        i+=1
        continue

    z.save(dimg)