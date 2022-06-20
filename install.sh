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
echo "****************** Download demo sequences ******************"
gdown https://drive.google.com/uc?id=1LeYyYxPA5XBF4m73T1p9qXsmGKVV7Fmm -O mfDiMP/demo/bus1.zip
gdown https://drive.google.com/uc?id=1yrtTYKcUfNsk-jxjXmY54FIxtBS3Fz9H -O mfDiMP/demo/bike_man.zip
gdown https://drive.google.com/uc?id=1uiqFuG3RnWcyu-OaRzKIQhrCLXKKM7_z -O mfDiMP/demo/car.zip
gdown https://drive.google.com/uc?id=1fL4Yo4xBSndeE_ShAqrraiHAL7QvrEAi -O mfDiMP/demo/night_car.zip

echo ""
echo ""
echo "****************** Installation complete! ******************"
