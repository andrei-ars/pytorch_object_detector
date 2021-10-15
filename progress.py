import progressbar
import time

for i in range(3):
    bar = progressbar.ProgressBar(maxval=100).start()
    t = 0
    while t <= 100:
        bar.update(t)
        time.sleep(0.02)
        t += 1
    bar.finish()