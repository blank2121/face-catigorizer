import face_recognition as fr
import itertools

def _has_face(image):
    # Load the image into memory
    image = face_recognition.load_image_file(image)
    # Detect the faces in the image
    face_locations = face_recognition.face_locations(image)
    # Return True if at least one face was detected
    return len(face_locations) > 0

def image_batch_generator(image_filenames, batch_size):
    # Create an iterator over the list of image filenames
    iterator = iter(image_filenames)
    while True:
        # Get the next batch of filenames
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        # Load and yield the images for this batch
        yield [load_image(filename) for filename in batch]
        

def face_filter(imgs: List[str]) -> List[str]:
    for img in image_batch_generator(imgs):
        if _has_face(img): continue
        imgs.remove(img)
    return imgs    