
def Image_Cut(src_path, out_path, raw_label, picture_size = 256):
    im = Image.open(src_path)
    size = min(im.width, im.height)
    raw_label[1] = im.height - raw_label[1]
    raw_label[3] = im.height - raw_label[3]
    im2 = im.crop((0, im.height - size, size, im.height))
    label= []
    for i in range(len(raw_label)):
        label.append(raw_label[i]/size)
    im2 = im2.resize((picture_size, picture_size))
    im2.save(out_path)
    label[1] = 1 - label[1]
    label[3] = 1 - label[3]
    return label
