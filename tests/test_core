"""Runs without TensorFlow: python -m unittest discover tests"""
import sys, types, unittest
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from emotion.config import EMOTIONS, CASCADE_PATH
from emotion.preprocessing import prepare_face, expand_box
from emotion.tracking import FaceTracker, iou, label_of
from emotion.face_detection import FaceDetector


class Core(unittest.TestCase):
    def test_prepare_face_shape_and_range(self):
        f = prepare_face(np.random.randint(0, 255, (120, 100), np.uint8))
        self.assertEqual(f.shape, (48, 48, 1)); self.assertEqual(f.dtype, np.float32)
        self.assertLessEqual(f.max(), 255)          # raw 0-255: model normalises internally

    def test_prepare_face_rejects_empty(self):
        self.assertIsNone(prepare_face(np.zeros((0, 0), np.uint8)))
        self.assertIsNone(prepare_face(np.zeros((4, 4), np.uint8)))

    def test_expand_box_clips(self):
        x, y, w, h = expand_box(0, 0, 50, 50, (60, 60, 3), 0.5)
        self.assertGreaterEqual(x, 0); self.assertLessEqual(x + w, 60); self.assertLessEqual(y + h, 60)

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 5, 5)), 0.0)

    def test_tracker_keeps_identity_and_expires(self):
        tr = FaceTracker(max_missed=2)
        a = tr.assign([(10, 10, 50, 50)])[0]; a.update((10, 10, 50, 50), np.eye(7)[3])
        b = tr.assign([(12, 11, 50, 50)])[0]
        self.assertEqual(a.tid, b.tid)
        for _ in range(4): tr.assign([])
        self.assertEqual(len(tr.tracks), 0)

    def test_smoothing_suppresses_flicker(self):
        tr = FaceTracker(); t = tr.assign([(0, 0, 40, 40)])[0]
        for i in range(9): t.update((0, 0, 40, 40), np.eye(7)[3])      # happy x9
        t.update((0, 0, 40, 40), np.eye(7)[5])                          # one sad frame
        self.assertEqual(label_of(t.smoothed())[0], "happy")

    def test_detector_loads(self):
        self.assertTrue(Path(CASCADE_PATH).exists())
        det = FaceDetector()
        self.assertEqual(det.detect(np.zeros((240, 320, 3), np.uint8)), [])

    def test_app_pipeline_with_stub_model(self):
        sys.modules.setdefault("keras", types.ModuleType("keras"))
        import app
        class Stub:
            path = Path("stub.keras")
            def predict_faces(self, crops):
                p = np.zeros((len(crops), 7), "float32"); p[:, 3] = 1; return p
        frame = np.full((240, 320, 3), 128, np.uint8)
        det = FaceDetector(); det.detect = lambda f, g=None: [(80, 60, 100, 100)]  # force one face
        n = app.process_frame(frame, det, Stub(), FaceTracker())
        self.assertEqual(n, 1)

if __name__ == "__main__":
    unittest.main()

