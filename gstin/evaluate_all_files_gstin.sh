while read a b
do
python3 /home/ritesh/Desktop/shopx2/image_processing_projects/Aadhaar\&PAN\ Detection/gstin/cropped_generator12.py --window $a --k $b >> output.txt
echo $a
done<"w_k.txt"
