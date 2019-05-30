import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
#Định dạng các loại ảnh
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Duyệt qua mỗi người trong thư mục TRAIN_DATA
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Tiếp tục duyệt qua từng tấm hình của mỗi người
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # Nếu không có người nào hoặc có quá nhiều người thì tấm hình này sẽ bị bỏ qua
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face"
                if len(face_bounding_boxes) < 1
                else "Found more than one face"))
            else:
                # Ngược lại tiến hành face encoding và thêm vào tập training
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # n_neighbor tự động sẽ được gán giá trị bất kì nếu không được truyền vào
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Training và tạo ra model classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Lưu lại model
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load model đã được trainning lên bộ nhớ
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load hình ảnh test lên và tìm vị trí các khuôn mặt trong bức hình
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # Trả về mảng rỗng nếu không tìm thấy khuôn mặt nào
    if len(X_face_locations) == 0:
        return []

    # Tiến hành face encoding các khuôn mặt tìm được
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Sử dụng model KNN để tìm ra khuôn mặt phù hợp nhất
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Trả về mảng danh sách vị trí các khuôn mặt và tên tương ứng
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec
            in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(image_file, predictions):
    img_path = os.path.join("test_data", image_file)
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Vẽ khung bao quanh các khuôn mặt
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Gán tên tương ứng lên từng khuôn mặt
        name = name.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 12, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw

    # Hiển thị hình ảnh chứa các khuôn mặt đã được nhận diện lên
    pil_image.show()

    # Lưu kết quả xuống ổ đĩa
    pil_image.save(os.path.join("RESULT_DATA", image_file))


if __name__ == "__main__":
    # Training model và lưu xuống ổ đĩa
    print("Training KNN classifier...")
    classifier = train("train_data", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Máy học hoàn thành!")

    # Sử dụng model đã train để dự đoán tên của các khuôn mặt có trong ảnh test
    for image_file in os.listdir("TEST_DATA"):
        full_file_path = os.path.join("TEST_DATA", image_file)

        print("Looking for faces in {}".format(image_file))

        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # In kết quả ra console
        count = 0
        for name, (top, right, bottom, left) in predictions:
            print("- Đã tìm thấy {}".format(name))
            if name != "Không tìm thấy":
                count += 1

        print("Recognized {}/{} faces found".format(count, len(predictions)))
        print("===============================")

        # Hiển thị ảnh chứa tên các khuôn mặt được nhận dạng
        show_prediction_labels_on_image(image_file, predictions)
