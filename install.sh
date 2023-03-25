echo "****************** Installing pytorch ******************"
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
pip install matplotlib

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python 

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Download model ******************"
gdown https://drive.google.com/uc?id=1vnof9qMFsfwmn8xk-UKaFhHTYM1F85j2 -O OptimTracker_ep0045.pth.tar

echo ""
echo ""
echo "****************** Installation complete! ******************"
