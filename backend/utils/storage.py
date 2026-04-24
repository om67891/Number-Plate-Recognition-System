from datetime import date

history = []


def save_plate(data):
    history.append(data)


def get_history():
    return history


def get_stats():
    data = get_history()
    valid_plates = [d["plate_text"] for d in data if d["plate_text"] != "UNREADABLE"]

    return {
        "total_vehicles": len(valid_plates),
        "unique_vehicles": len(set(valid_plates)),
    }


def vehicles_today():
    today = str(date.today())
    data = get_history()

    count = sum(1 for d in data if today in d["timestamp"])

    return {"vehicles_today": count}
