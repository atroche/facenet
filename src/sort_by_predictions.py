import csv
from PIL import Image

rows = csv.reader(open("scores_for_unlabelled.csv", "r"))
rows = sorted(rows, key=lambda x:x[2], reverse=True)[1:]
[row[2] for row in rows[:10]]
rows[0]

for index, row in enumerate(rows):
    image = Image.open(open(row[-1], "rb"))
    image.save("datasets/videos/faces/sorted/harvard_commencement/%s.jpg" % index)

