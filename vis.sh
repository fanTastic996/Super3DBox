
#!/bin/bash


cd training
tensorboard --logdir=./logs --port=6006 --reload_multifile false 