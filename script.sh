
input="D:/TGMT/TGMT/Lab01/LAB01/b1.jpg"

grayoutput="D:/TGMT/TGMT/Lab01/LAB01/output_img/gray.jpg"

brightoutput="D:/TGMT/TGMT/Lab01/LAB01/output_img/bright.jpg"

contrastoutput="D:/TGMT/TGMT/Lab01/LAB01/output_img/contrast.jpg"

filteroutput="D:/TGMT/TGMT/Lab01/LAB01/output_img/filter.jpg"

edgeoutput="D:/TGMT/TGMT/Lab01/LAB01/output_img/edge.jpg"

# Chạy chương trình tương ứng với tham số truyền vào
if [ $1 == "rgb2gray.py" ]
then
    python rgb2gray.py --input $input --output $grayoutput
elif [ $1 == "brightness.py" ]
then
    bias="50"
    python brightness.py --input $input --output $brightoutput --brightness $bias
elif [ $1 == "contrast.py" ]
then
    alpha="100"
    python contrast.py --input $input --output $contrastoutput --contrast $alpha
elif [ $1 == "filter.py" ]
then
    filter="avg"
    kernel_size="3"
    python filter.py --input $input --output $filteroutput --filter $filter --kernel_size $kernel_size
elif [ $1 == "edge.py" ]
then
    edge="laplace"
    size="5"
    python edge.py --input $input --output $edgeoutput --edge $edge --size $size
fi