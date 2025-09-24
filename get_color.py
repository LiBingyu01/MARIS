import json

def merge_categories(list1, list2):
    """
    合并两个类别列表，根据id求并集（保留每个id的第一个出现的记录）
    """
    # 使用字典存储，key为id，确保每个id只保留一个
    merged = {}
    
    # 先添加第一个列表的元素
    for item in list1:
        item_id = item.get('id')
        if item_id not in merged:
            merged[item_id] = item
    
    # 再添加第二个列表的元素，不覆盖已存在的id
    for item in list2:
        item_id = item.get('id')
        if item_id not in merged:
            merged[item_id] = item
    
    # 按id排序并转换为列表
    sorted_merged = sorted(merged.values(), key=lambda x: x.get('id'))
    return sorted_merged

def extract_name_and_color(categories):
    """提取名称和颜色信息，返回分离的列表"""
    classes = []
    palette = []
    
    for item in categories:
        classes.append(item.get('name'))
        palette.append(item.get('color'))
        
    return classes, palette

def save_classes(classes, filename='classes.txt'):
    """保存类别列表到文件，使用指定格式"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('classes = [\n    ')
        # 处理列表元素，添加引号和逗号
        class_strings = [f"'{cls}'" for cls in classes]
        # 每6个元素换行，保持格式整洁
        for i, cls_str in enumerate(class_strings):
            if i > 0 and i % 6 == 0:
                f.write(',\n    ')
            elif i > 0:
                f.write(', ')
            f.write(cls_str)
        f.write('\n]\n')

def save_palette(palette, filename='palette.txt'):
    """保存颜色列表到文件，使用指定格式"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('palette = [\n    ')
        # 处理颜色列表，转换为字符串
        palette_strings = [str(color) for color in palette]
        # 每5个元素换行，保持格式整洁
        for i, pal_str in enumerate(palette_strings):
            if i > 0 and i % 5 == 0:
                f.write(',\n    ')
            elif i > 0:
                f.write(', ')
            f.write(pal_str)
        f.write('\n]\n')


# 示例数据 - 请替换为你的实际数据
# 第一个类别列表
WATEROVS_CATEGORIES_TRAIN = [
{"color": [255, 0, 0], "isthing": 1, "id": 1, "name": "Diver"},
{"color": [0, 255, 0], "isthing": 1, "id": 2, "name": "Swimmer"},
{"color": [0, 0, 255], "isthing": 1, "id": 3, "name": "Geoduck"},
{"color": [255, 255, 0], "isthing": 1, "id": 4, "name": "Linckia Laevigata"},
{"color": [255, 0, 255], "isthing": 1, "id": 5, "name": "Manta Ray"},
{"color": [128, 0, 0], "isthing": 1, "id": 7, "name": "Sawfish"},
{"color": [0, 128, 0], "isthing": 1, "id": 8, "name": "Bullhead Shark"},
{"color": [128, 128, 0], "isthing": 1, "id": 10, "name": "Whale Shark"},
{"color": [128, 0, 128], "isthing": 1, "id": 11, "name": "Hammerhead Shark"},
{"color": [0, 0, 192], "isthing": 1, "id": 15, "name": "Moray Eel"},
{"color": [192, 192, 0], "isthing": 1, "id": 16, "name": "Orbicular Batfish"},
{"color": [192, 0, 192], "isthing": 1, "id": 17, "name": "Lionfish"},
{"color": [0, 192, 192], "isthing": 1, "id": 18, "name": "Trumpetfish"},
{"color": [64, 64, 0], "isthing": 1, "id": 22, "name": "Enoplosus Armatus"},
{"color": [0, 64, 64], "isthing": 1, "id": 24, "name": "Mola"},
{"color": [255, 128, 0], "isthing": 1, "id": 25, "name": "Moorish Idol"},
{"color": [255, 0, 128], "isthing": 1, "id": 26, "name": "Bicolor Angelfish"},
{"color": [255, 64, 0], "isthing": 1, "id": 31, "name": "Redsea Bannerfish"},
{"color": [0, 64, 255], "isthing": 1, "id": 36, "name": "Twin-spot Goby"},
{"color": [0, 255, 192], "isthing": 1, "id": 40, "name": "Blue Parrotfish"},
{"color": [192, 0, 255], "isthing": 1, "id": 41, "name": "Stoplight Parrotfish"},
{"color": [0, 128, 64], "isthing": 1, "id": 46, "name": "Red Sea Clownfish"},
{"color": [255, 255, 128], "isthing": 1, "id": 49, "name": "Giant Wrasse"},
{"color": [255, 128, 255], "isthing": 1, "id": 50, "name": "Spotted Wrasse"},
{"color": [128, 255, 255], "isthing": 1, "id": 51, "name": "Anampses Twistii"},
{"color": [255, 128, 128], "isthing": 1, "id": 52, "name": "Blue-spotted Wrasse"},
{"color": [128, 64, 255], "isthing": 1, "id": 56, "name": "Potato Grouper"},
{"color": [64, 128, 255], "isthing": 1, "id": 57, "name": "Graysby"},
{"color": [64, 192, 0], "isthing": 1, "id": 63, "name": "Whitespotted Surgeonfish"},
{"color": [192, 128, 0], "isthing": 1, "id": 67, "name": "Regal Blue Tang"},
{"color": [192, 0, 128], "isthing": 1, "id": 68, "name": "Lined Surgeonfish"},
{"color": [128, 192, 0], "isthing": 1, "id": 69, "name": "Achilles Tang"},
{"color": [0, 128, 192], "isthing": 1, "id": 72, "name": "Saddle Butterflyfish"},
{"color": [192, 64, 128], "isthing": 1, "id": 73, "name": "Mirror Butterflyfish"},
{"color": [128, 64, 192], "isthing": 1, "id": 74, "name": "Bluecheek Butterflyfish"},
{"color": [128, 192, 64], "isthing": 1, "id": 77, "name": "Threadfin Butterflyfish"},
{"color": [17, 100, 60], "isthing": 1, "id": 81, "name": "Giant Clams"},
{"color": [101, 230, 195], "isthing": 1, "id": 82, "name": "Scallop"},
{"color": [34, 184, 92], "isthing": 1, "id": 85, "name": "Nautilus"},
{"color": [52, 46, 241], "isthing": 1, "id": 86, "name": "Triton's Trumpet"},
{"color": [251, 202, 202], "isthing": 1, "id": 90, "name": "Common Octopus"},
{"color": [209, 69, 192], "isthing": 1, "id": 91, "name": "Squid"},
{"color": [193, 6, 44], "isthing": 1, "id": 92, "name": "Cuttlefish"},
{"color": [153, 63, 18], "isthing": 1, "id": 93, "name": "Sea Anemone"},
{"color": [237, 54, 96], "isthing": 1, "id": 94, "name": "Lion's Mane Jellyfish"},
{"color": [158, 183, 109], "isthing": 1, "id": 95, "name": "Moon Jellyfish"},
{"color": [63, 156, 226], "isthing": 1, "id": 96, "name": "Fried Egg Jellyfish"},
{"color": [5, 27, 48], "isthing": 1, "id": 97, "name": "Fan Coral"},
{"color": [142, 142, 165], "isthing": 1, "id": 98, "name": "Elkhorn Coral"},
{"color": [92, 85, 9], "isthing": 1, "id": 99, "name": "Brain Coral"},
{"color": [58, 73, 38], "isthing": 1, "id": 100, "name": "Sea Urchin"},
{"color": [205, 207, 4], "isthing": 1, "id": 101, "name": "Sea Cucumber"},
{"color": [98, 221, 109], "isthing": 1, "id": 102, "name": "Crinoid"},
{"color": [191, 232, 250], "isthing": 1, "id": 103, "name": "Oreaster Reticulatus"},
{"color": [199, 155, 140], "isthing": 1, "id": 104, "name": "Protoreaster Nodosus"},
{"color": [97, 165, 57], "isthing": 1, "id": 105, "name": "Killer Whale"},
{"color": [62, 213, 175], "isthing": 1, "id": 109, "name": "Manatee"},
{"color": [69, 93, 203], "isthing": 1, "id": 110, "name": "Sea Lion"},
{"color": [245, 95, 53], "isthing": 1, "id": 111, "name": "Dolphin"},
{"color": [28, 51, 88], "isthing": 1, "id": 112, "name": "Walrus"},
{"color": [198, 245, 70], "isthing": 1, "id": 113, "name": "Dugong"},
{"color": [21, 119, 187], "isthing": 1, "id": 114, "name": "Turtle"},
{"color": [241, 56, 102], "isthing": 1, "id": 115, "name": "Snake"},
{"color": [10, 106, 191], "isthing": 1, "id": 117, "name": "Spiny Lobster"},
{"color": [51, 21, 76], "isthing": 1, "id": 118, "name": "Common Prawn"},
{"color": [86, 98, 246], "isthing": 1, "id": 119, "name": "Mantis Shrimp"},
{"color": [56, 20, 155], "isthing": 1, "id": 120, "name": "King Crab"},
{"color": [9, 64, 12], "isthing": 1, "id": 127, "name": "Plastic Bag"},
{"color": [51, 181, 44], "isthing": 1, "id": 132, "name": "Surgical Mask"},
{"color": [57, 112, 184], "isthing": 1, "id": 133, "name": "Tyre"},
{"color": [252, 88, 148], "isthing": 1, "id": 134, "name": "Can"},
{"color": [86, 5, 179], "isthing": 1, "id": 135, "name": "Shipwreck"},
{"color": [198, 144, 220], "isthing": 1, "id": 136, "name": "Wrecked Aircraft"},
{"color": [127, 245, 183], "isthing": 1, "id": 139, "name": "Gun"},
{"color": [118, 73, 129], "isthing": 1, "id": 140, "name": "Phone"},
{"color": [151, 16, 216], "isthing": 1, "id": 144, "name": "Coin"},
{"color": [248, 56, 200], "isthing": 1, "id": 145, "name": "Statue"},
{"color": [34, 6, 104], "isthing": 1, "id": 146, "name": "Amphora"},
{"color": [239, 63, 228], "isthing": 1, "id": 149, "name": "Autonomous Underwater Vehicle (AUV)"},
{"color": [153, 236, 17], "isthing": 1, "id": 150, "name": "Remotely Operated Vehicle (ROV)"},
{"color": [215, 83, 4], "isthing": 1, "id": 152, "name": "Personal Submarines"},
{"color": [162, 70, 182], "isthing": 1, "id": 153, "name": "Ship's Anode"},
{"color": [8, 177, 233], "isthing": 1, "id": 154, "name": "Over Board Valve"},
{"color": [87, 8, 70], "isthing": 1, "id": 155, "name": "Propeller"},
]



WATEROVS_CATEGORIES_VAL = [
{"color": [255, 0, 0], "isthing": 1, "id": 1, "name": "Diver"},
{"color": [0, 255, 0], "isthing": 1, "id": 2, "name": "Swimmer"},
{"color": [0, 0, 255], "isthing": 1, "id": 3, "name": "Geoduck"},
{"color": [255, 255, 0], "isthing": 1, "id": 4, "name": "Linckia Laevigata"},
{"color": [0, 255, 255], "isthing": 1, "id": 6, "name": "Electric Ray"},
{"color": [0, 0, 128], "isthing": 1, "id": 9, "name": "Great White Shark"},
{"color": [128, 128, 0], "isthing": 1, "id": 10, "name": "Whale Shark"},
{"color": [128, 0, 128], "isthing": 1, "id": 11, "name": "Hammerhead Shark"},
{"color": [0, 128, 128], "isthing": 1, "id": 12, "name": "Thresher Shark"},
{"color": [192, 0, 0], "isthing": 1, "id": 13, "name": "Sea Dragon"},
{"color": [0, 192, 0], "isthing": 1, "id": 14, "name": "Hippocampus"},
{"color": [0, 0, 192], "isthing": 1, "id": 15, "name": "Moray Eel"},
{"color": [192, 0, 192], "isthing": 1, "id": 17, "name": "Lionfish"},
{"color": [0, 192, 192], "isthing": 1, "id": 18, "name": "Trumpetfish"},
{"color": [64, 0, 0], "isthing": 1, "id": 19, "name": "Flounder"},
{"color": [0, 64, 0], "isthing": 1, "id": 20, "name": "Frogfish"},
{"color": [0, 0, 64], "isthing": 1, "id": 21, "name": "Sailfish"},
{"color": [64, 64, 0], "isthing": 1, "id": 22, "name": "Enoplosus Armatus"},
{"color": [64, 0, 64], "isthing": 1, "id": 23, "name": "Pseudanthias Pleurotaenia"},
{"color": [128, 255, 0], "isthing": 1, "id": 27, "name": "Atlantic Spadefish"},
{"color": [0, 255, 128], "isthing": 1, "id": 28, "name": "Spotted Drum"},
{"color": [128, 0, 255], "isthing": 1, "id": 29, "name": "Threespot Angelfish"},
{"color": [0, 128, 255], "isthing": 1, "id": 30, "name": "Chromis Dimidiata"},
{"color": [255, 64, 0], "isthing": 1, "id": 31, "name": "Redsea Bannerfish"},
{"color": [255, 0, 64], "isthing": 1, "id": 32, "name": "Heniochus Varius"},
{"color": [64, 255, 0], "isthing": 1, "id": 33, "name": "Maldives Damselfish"},
{"color": [0, 255, 64], "isthing": 1, "id": 34, "name": "Scissortail Sergeant"},
{"color": [64, 0, 255], "isthing": 1, "id": 35, "name": "Fire Goby"},
{"color": [255, 192, 0], "isthing": 1, "id": 37, "name": "Porcupinefish"},
{"color": [255, 0, 192], "isthing": 1, "id": 38, "name": "Yellow Boxfish"},
{"color": [192, 255, 0], "isthing": 1, "id": 39, "name": "Blackspotted Puffer"},
{"color": [0, 255, 192], "isthing": 1, "id": 40, "name": "Blue Parrotfish"},
{"color": [0, 192, 255], "isthing": 1, "id": 42, "name": "Pomacentrus Sulfureus"},
{"color": [128, 64, 0], "isthing": 1, "id": 43, "name": "Lunar Fusilier"},
{"color": [128, 0, 64], "isthing": 1, "id": 44, "name": "Ocellaris Clownfish"},
{"color": [64, 128, 0], "isthing": 1, "id": 45, "name": "Cinnamon Clownfish"},
{"color": [64, 0, 128], "isthing": 1, "id": 47, "name": "Pink Anemonefish"},
{"color": [0, 64, 128], "isthing": 1, "id": 48, "name": "Orange Skunk Clownfish"},
{"color": [255, 255, 128], "isthing": 1, "id": 49, "name": "Giant Wrasse"},
{"color": [255, 128, 128], "isthing": 1, "id": 52, "name": "Blue-spotted Wrasse"},
{"color": [128, 255, 128], "isthing": 1, "id": 53, "name": "Slingjaw Wrasse"},
{"color": [128, 128, 255], "isthing": 1, "id": 54, "name": "Red-breasted Wrasse"},
{"color": [255, 64, 128], "isthing": 1, "id": 55, "name": "Peacock Grouper"},
{"color": [64, 255, 128], "isthing": 1, "id": 58, "name": "Redmouth Grouper"},
{"color": [128, 255, 64], "isthing": 1, "id": 59, "name": "Humpback Grouper"},
{"color": [255, 128, 64], "isthing": 1, "id": 60, "name": "Coral Hind"},
{"color": [192, 64, 0], "isthing": 1, "id": 61, "name": "Porkfish"},
{"color": [192, 0, 64], "isthing": 1, "id": 62, "name": "Anyperodon Leucogrammicus"},
{"color": [0, 192, 64], "isthing": 1, "id": 64, "name": "Orange-band Surgeonfish"},
{"color": [64, 0, 192], "isthing": 1, "id": 65, "name": "Convict Surgeonfish"},
{"color": [0, 64, 192], "isthing": 1, "id": 66, "name": "Sohal Surgeonfish"},
{"color": [0, 192, 128], "isthing": 1, "id": 70, "name": "Powder Blue Tang"},
{"color": [128, 0, 192], "isthing": 1, "id": 71, "name": "Whitecheek Surgeonfish"},
{"color": [128, 64, 192], "isthing": 1, "id": 74, "name": "Bluecheek Butterflyfish"},
{"color": [64, 192, 128], "isthing": 1, "id": 75, "name": "Blacktail Butterflyfish"},
{"color": [64, 128, 192], "isthing": 1, "id": 76, "name": "Raccoon Butterflyfish"},
{"color": [128, 192, 64], "isthing": 1, "id": 77, "name": "Threadfin Butterflyfish"},
{"color": [192, 128, 64], "isthing": 1, "id": 78, "name": "Eritrean Butterflyfish"},
{"color": [255, 192, 64], "isthing": 1, "id": 79, "name": "Pyramid Butterflyfish"},
{"color": [192, 255, 64], "isthing": 1, "id": 80, "name": "Copperband Butterflyfish"},
{"color": [17, 100, 60], "isthing": 1, "id": 81, "name": "Giant Clams"},
{"color": [230, 222, 163], "isthing": 1, "id": 83, "name": "Abalone"},
{"color": [116, 140, 146], "isthing": 1, "id": 84, "name": "Queen Conch"},
{"color": [34, 184, 92], "isthing": 1, "id": 85, "name": "Nautilus"},
{"color": [52, 46, 241], "isthing": 1, "id": 86, "name": "Triton's Trumpet"},
{"color": [128, 138, 10], "isthing": 1, "id": 87, "name": "Sea Slug"},
{"color": [133, 120, 39], "isthing": 1, "id": 88, "name": "Dumbo Octopus"},
{"color": [34, 212, 240], "isthing": 1, "id": 89, "name": "Blue-ringed Octopus"},
{"color": [158, 183, 109], "isthing": 1, "id": 95, "name": "Moon Jellyfish"},
{"color": [63, 156, 226], "isthing": 1, "id": 96, "name": "Fried Egg Jellyfish"},
{"color": [5, 27, 48], "isthing": 1, "id": 97, "name": "Fan Coral"},
{"color": [142, 142, 165], "isthing": 1, "id": 98, "name": "Elkhorn Coral"},
{"color": [92, 85, 9], "isthing": 1, "id": 99, "name": "Brain Coral"},
{"color": [58, 73, 38], "isthing": 1, "id": 100, "name": "Sea Urchin"},
{"color": [191, 232, 250], "isthing": 1, "id": 103, "name": "Oreaster Reticulatus"},
{"color": [199, 155, 140], "isthing": 1, "id": 104, "name": "Protoreaster Nodosus"},
{"color": [76, 242, 238], "isthing": 1, "id": 106, "name": "Sperm Whale"},
{"color": [93, 206, 247], "isthing": 1, "id": 107, "name": "Humpback Whale"},
{"color": [102, 160, 133], "isthing": 1, "id": 108, "name": "Seal"},
{"color": [62, 213, 175], "isthing": 1, "id": 109, "name": "Manatee"},
{"color": [69, 93, 203], "isthing": 1, "id": 110, "name": "Sea Lion"},
{"color": [245, 95, 53], "isthing": 1, "id": 111, "name": "Dolphin"},
{"color": [28, 51, 88], "isthing": 1, "id": 112, "name": "Walrus"},
{"color": [198, 245, 70], "isthing": 1, "id": 113, "name": "Dugong"},
{"color": [21, 119, 187], "isthing": 1, "id": 114, "name": "Turtle"},
{"color": [241, 56, 102], "isthing": 1, "id": 115, "name": "Snake"},
{"color": [193, 239, 38], "isthing": 1, "id": 116, "name": "Homarus"},
{"color": [0, 196, 214], "isthing": 1, "id": 121, "name": "Hermit Crab"},
{"color": [94, 175, 14], "isthing": 1, "id": 122, "name": "Cancer Pagurus"},
{"color": [31, 97, 216], "isthing": 1, "id": 123, "name": "Swimming Crab"},
{"color": [24, 30, 177], "isthing": 1, "id": 124, "name": "Spanner Crab"},
{"color": [2, 204, 158], "isthing": 1, "id": 125, "name": "Penguin"},
{"color": [5, 214, 252], "isthing": 1, "id": 126, "name": "Sponge"},
{"color": [9, 64, 12], "isthing": 1, "id": 127, "name": "Plastic Bag"},
{"color": [166, 189, 121], "isthing": 1, "id": 128, "name": "Plastic Bottle"},
{"color": [129, 25, 8], "isthing": 1, "id": 129, "name": "Plastic Cup"},
{"color": [53, 114, 0], "isthing": 1, "id": 130, "name": "Plastic Box"},
{"color": [83, 49, 65], "isthing": 1, "id": 131, "name": "Glass Bottle"},
{"color": [86, 5, 179], "isthing": 1, "id": 135, "name": "Shipwreck"},
{"color": [198, 144, 220], "isthing": 1, "id": 136, "name": "Wrecked Aircraft"},
{"color": [188, 77, 97], "isthing": 1, "id": 137, "name": "Wrecked Car"},
{"color": [248, 240, 45], "isthing": 1, "id": 138, "name": "Wrecked Tank"},
{"color": [248, 191, 96], "isthing": 1, "id": 141, "name": "Ring"},
{"color": [4, 245, 60], "isthing": 1, "id": 142, "name": "Boots"},
{"color": [215, 9, 173], "isthing": 1, "id": 143, "name": "Glasses"},
{"color": [248, 56, 200], "isthing": 1, "id": 145, "name": "Statue"},
{"color": [34, 6, 104], "isthing": 1, "id": 146, "name": "Amphora"},
{"color": [11, 91, 206], "isthing": 1, "id": 147, "name": "Anchor"},
{"color": [0, 143, 25], "isthing": 1, "id": 148, "name": "Ship's Wheel"},
{"color": [153, 236, 17], "isthing": 1, "id": 150, "name": "Remotely Operated Vehicle (ROV)"},
{"color": [28, 135, 172], "isthing": 1, "id": 151, "name": "Military Submarines"},
{"color": [87, 8, 70], "isthing": 1, "id": 155, "name": "Propeller"},
{"color": [54, 92, 117], "isthing": 1, "id": 156, "name": "Sea Chest Grating"},
{"color": [206, 15, 125], "isthing": 1, "id": 157, "name": "Submarine Pipeline"},
{"color": [10, 246, 2], "isthing": 1, "id": 158, "name": "Pipeline's Anode"}]

if __name__ == "__main__":
    # 合并两个列表
    merged = merge_categories(WATEROVS_CATEGORIES_TRAIN, WATEROVS_CATEGORIES_VAL)
    
    # 提取名称和颜色，得到两个分离的列表
    classes, palette = extract_name_and_color(merged)
    
    # 分别保存到文件
    save_classes(classes)
    save_palette(palette)
    
    print(f"合并完成，共 {len(classes)} 个类别")
    print("类别已保存到 classes.txt")
    print("颜色已保存到 palette.txt")
    
