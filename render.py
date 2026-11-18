import cv2


def draw_detection(frame, people_count, boxes, crowd_limit):
    crowd_detected = people_count > crowd_limit
    color = (0, 0, 255) if crowd_detected else (0, 255, 0)
    status = "CROWD ALERT" if crowd_detected else "NORMAL"

    for x1, y1, x2, y2, confidence in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"Person {confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    cv2.putText(
        frame,
        f"People: {people_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        frame,
        status,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return crowd_detected, status


def draw_message(frame, message, position_y, color=(0, 255, 255)):
    cv2.putText(
        frame,
        message,
        (10, position_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
