

import tqdm
import time

progress = tqdm.tqdm(total=100, desc='Progress')

for i in range(100):

    time.sleep(0.1)
    progress.update(1)

progress.close()

print('done!')