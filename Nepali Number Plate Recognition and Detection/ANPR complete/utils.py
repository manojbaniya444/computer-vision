from db import fetch_latest_records

def get_license_plate_coordinates(results):
        max_confidence = 0
        best_coordinates = None
        class_names = ["car","license_plate","bike","bus"]

        for result in results:
            boxes = result.boxes.xyxy.cpu()
            clss = result.boxes.cls.cpu().tolist()
            confs = result.boxes.conf.float().cpu().tolist()

            for box, cls, conf in zip(boxes, clss, confs):
                class_name = class_names[int(cls)]

                if class_name == 'license_plate' and conf > 0.5:
                    if conf > max_confidence:
                        max_confidence = conf
                        best_coordinates = box

        return best_coordinates
    
##? Update table
# Fetch and insert latest records into Treeview
def update_records(tree):
    latest_records = fetch_latest_records()
    tree.delete(*tree.get_children())
    for _, record in enumerate(latest_records):  
        tree.insert("", "end", values=record, tags=("row",))
        tree.insert("", "end", values=("", "", ""), tags=("spacer",))
