import pandas as pd

# 读取txt文件（detectron2输出）
txt_file = "results.txt"

# 把表格转为DataFrame
import pandas as pd

# 读取txt文件（detectron2输出）
txt_file = "results.txt"

# 用于存储解析后的数据
data = []

with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        # 按竖线分割并去除空白
        parts = [x.strip() for x in line.split("|") if x.strip()]
        # 每行有3列（category, AP, AP50, AP75），需确保至少有4个部分（包含分隔和三列数据）
        if len(parts) >= 4:
            category = parts[0]
            try:
                ap = float(parts[1])
                ap50 = float(parts[2])
                ap75 = float(parts[3])
                # 将三列数据存入列表
                data.append([category, ap, ap50, ap75])
            except ValueError:
                # 若转换失败，打印错误信息并跳过该行
                print(f"转换数值失败，跳过该行：{line}")

# 转换为DataFrame并指定列名
df = pd.DataFrame(data, columns=["category", "AP", "AP50", "AP75"])

# 打印DataFrame
print(df)

# 定义类别集合

intersection_classes = ['Amphora', 'Blue Parrotfish', 'Blue-spotted Wrasse', 'Bluecheek Butterflyfish',
    'Brain Coral', 'Diver', 'Dolphin', 'Dugong', 'Elkhorn Coral', 'Enoplosus Armatus',
    'Fan Coral', 'Fried Egg Jellyfish', 'Geoduck', 'Giant Clams', 'Giant Wrasse',
    'Hammerhead Shark', 'Linckia Laevigata', 'Lionfish', 'Manatee', 'Moon Jellyfish',
    'Moray Eel', 'Nautilus', 'Oreaster Reticulatus', 'Plastic Bag', 'Propeller',
    'Protoreaster Nodosus', 'Redsea Bannerfish', 'Remotely Operated Vehicle (ROV)',
    'Sea Lion', 'Sea Urchin', 'Shipwreck', 'Snake', 'Statue', 'Swimmer',
    'Threadfin Butterflyfish', "Triton's Trumpet", 'Trumpetfish', 'Turtle',
    'Walrus', 'Whale Shark', 'Wrecked Aircraft']

ov_classes = ['Abalone', 'Anchor', 'Anyperodon Leucogrammicus', 'Atlantic Spadefish',
    'Blackspotted Puffer', 'Blacktail Butterflyfish', 'Blue-ringed Octopus', 'Boots',
    'Cancer Pagurus', 'Chromis Dimidiata', 'Cinnamon Clownfish', 'Convict Surgeonfish',
    'Copperband Butterflyfish', 'Coral Hind', 'Dumbo Octopus', 'Electric Ray',
    'Eritrean Butterflyfish', 'Fire Goby', 'Flounder', 'Frogfish', 'Glass Bottle',
    'Glasses', 'Great White Shark', 'Heniochus Varius', 'Hermit Crab', 'Hippocampus',
    'Homarus', 'Humpback Grouper', 'Humpback Whale', 'Lunar Fusilier',
    'Maldives Damselfish', 'Military Submarines', 'Ocellaris Clownfish',
    'Orange Skunk Clownfish', 'Orange-band Surgeonfish', 'Peacock Grouper', 'Penguin',
    'Pink Anemonefish', "Pipeline's Anode", 'Plastic Bottle', 'Plastic Box',
    'Plastic Cup', 'Pomacentrus Sulfureus', 'Porcupinefish', 'Porkfish',
    'Powder Blue Tang', 'Pseudanthias Pleurotaenia', 'Pyramid Butterflyfish',
    'Queen Conch', 'Raccoon Butterflyfish', 'Red-breasted Wrasse', 'Redmouth Grouper',
    'Ring', 'Sailfish', 'Scissortail Sergeant', 'Sea Chest Grating', 'Sea Dragon',
    'Sea Slug', 'Seal', "Ship's Wheel", 'Slingjaw Wrasse', 'Sohal Surgeonfish',
    'Spanner Crab', 'Sperm Whale', 'Sponge', 'Spotted Drum', 'Submarine Pipeline',
    'Swimming Crab', 'Threespot Angelfish', 'Thresher Shark', 'Whitecheek Surgeonfish',
    'Wrecked Car', 'Wrecked Tank', 'Yellow Boxfish']

all_classes = [
    "Diver", "Swimmer", "Geoduck", "Linckia Laevigata", "Manta Ray", "Electric Ray",
    "Sawfish", "Bullhead Shark", "Great White Shark", "Whale Shark", "Hammerhead Shark", "Thresher Shark",
    "Sea Dragon", "Hippocampus", "Moray Eel", "Orbicular Batfish", "Lionfish", "Trumpetfish",
    "Flounder", "Frogfish", "Sailfish", "Enoplosus Armatus", "Pseudanthias Pleurotaenia", "Mola",
    "Moorish Idol", "Bicolor Angelfish", "Atlantic Spadefish", "Spotted Drum", "Threespot Angelfish", "Chromis Dimidiata",
    "Redsea Bannerfish", "Heniochus Varius", "Maldives Damselfish", "Scissortail Sergeant", "Fire Goby", "Twin-spot Goby",
    "Porcupinefish", "Yellow Boxfish", "Blackspotted Puffer", "Blue Parrotfish", "Stoplight Parrotfish", "Pomacentrus Sulfureus",
    "Lunar Fusilier", "Ocellaris Clownfish", "Cinnamon Clownfish", "Red Sea Clownfish", "Pink Anemonefish", "Orange Skunk Clownfish",
    "Giant Wrasse", "Spotted Wrasse", "Anampses Twistii", "Blue-spotted Wrasse", "Slingjaw Wrasse", "Red-breasted Wrasse",
    "Peacock Grouper", "Potato Grouper", "Graysby", "Redmouth Grouper", "Humpback Grouper", "Coral Hind",
    "Porkfish", "Anyperodon Leucogrammicus", "Whitespotted Surgeonfish", "Orange-band Surgeonfish", "Convict Surgeonfish", "Sohal Surgeonfish",
    "Regal Blue Tang", "Lined Surgeonfish", "Achilles Tang", "Powder Blue Tang", "Whitecheek Surgeonfish", "Saddle Butterflyfish",
    "Mirror Butterflyfish", "Bluecheek Butterflyfish", "Blacktail Butterflyfish", "Raccoon Butterflyfish", "Threadfin Butterflyfish", "Eritrean Butterflyfish",
    "Pyramid Butterflyfish", "Copperband Butterflyfish", "Giant Clams", "Scallop", "Abalone", "Queen Conch",
    "Nautilus", "Triton's Trumpet", "Sea Slug", "Dumbo Octopus", "Blue-ringed Octopus", "Common Octopus",
    "Squid", "Cuttlefish", "Sea Anemone", "Lion's Mane Jellyfish", "Moon Jellyfish", "Fried Egg Jellyfish",
    "Fan Coral", "Elkhorn Coral", "Brain Coral", "Sea Urchin", "Sea Cucumber", "Crinoid",
    "Oreaster Reticulatus", "Protoreaster Nodosus", "Killer Whale", "Sperm Whale", "Humpback Whale", "Seal",
    "Manatee", "Sea Lion", "Dolphin", "Walrus", "Dugong", "Turtle",
    "Snake", "Homarus", "Spiny Lobster", "Common Prawn", "Mantis Shrimp", "King Crab",
    "Hermit Crab", "Cancer Pagurus", "Swimming Crab", "Spanner Crab", "Penguin", "Sponge",
    "Plastic Bag", "Plastic Bottle", "Plastic Cup", "Plastic Box", "Glass Bottle", "Surgical Mask",
    "Tyre", "Can", "Shipwreck", "Wrecked Aircraft", "Wrecked Car", "Wrecked Tank",
    "Gun", "Phone", "Ring", "Boots", "Glasses", "Coin",
    "Statue", "Amphora", "Anchor", "Ship's Wheel", "Autonomous Underwater Vehicle (AUV)", "Remotely Operated Vehicle (ROV)",
    "Military Submarines", "Personal Submarines", "Ship's Anode", "Over Board Valve", "Propeller", "Sea Chest Grating",
    "Submarine Pipeline", "Pipeline's Anode"
]

# 筛选
df_intersection = df[df['category'].isin(intersection_classes)]
df_ov = df[df['category'].isin(ov_classes)]
df_all = df[df['category'].isin(all_classes)]

print(f"df_all_{df}")
# 计算均值
intersection_mean_ap = df_intersection['AP'].mean()
ov_mean_ap = df_ov['AP'].mean()
all_mean_ap = df_all['AP'].mean()

intersection_mean_ap50 = df_intersection['AP50'].mean()
ov_mean_ap50 = df_ov['AP50'].mean()
all_mean_ap50 = df_all['AP50'].mean()

intersection_mean_ap75 = df_intersection['AP75'].mean()
ov_mean_ap75 = df_ov['AP75'].mean()
all_mean_ap75 = df_all['AP75'].mean()

print("\n")
print('Intersection Class:')
print("Intersection Class mAP:", intersection_mean_ap)
print("Intersection Class mAP50:", intersection_mean_ap50)
print("Intersection Class mAP75:", intersection_mean_ap75)

print("\n")
print('Open-Vocabulary Class :')
print("OV Class mAP:", ov_mean_ap)
print("OV Class mAP50:", ov_mean_ap50)
print("OV Class mAP75:", ov_mean_ap75)

print("\n")
print('Total:')
print("ALL Class mAP:", all_mean_ap)
print("ALL Class mAP50:", all_mean_ap50)
print("ALL Class mAP75:", all_mean_ap75)