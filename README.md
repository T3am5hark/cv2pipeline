# cv2pipeline
Image Processing Pipelines built on OpenCV2

____________
Installation
____________

___________________
Virtual Environment
___________________

run:

  sudo apt-get install virualenv

create:

  mkdir envs
  cd envs
  virtualenv -p python3 cv2pipeline

If you place the envs folder one level up from the project root,
you can activate from the project root directory:

  . activate.sh

________
Package Dependencies
________

After creating a virtual environment, install the dependencies:

pip install -r requirements.txt

for RPi4, use:

pip install -r requirements.pi4.txt

You may need to install the following:

sudo apt-get install libatlas-base-dev
