package com.example.dutstudenttracker;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;
import java.util.Arrays;

import androidx.annotation.OptIn;
import androidx.camera.core.ExperimentalGetImage;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;



public class CameraActivity extends AppCompatActivity {

    private FaceOverlayView faceOverlayView;
    private PreviewView previewView;
    private ExecutorService cameraExecutor;
    private FaceDetector faceDetector;
    private TfLiteFaceEmbedder faceEmbedder;
    private String lastFeedback = "";
    private long lastToastTime = 0;



    @Override
    @ExperimentalGetImage
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            faceEmbedder = new TfLiteFaceEmbedder(this, "facenet.tflite");
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to load face recognition model", Toast.LENGTH_LONG).show();
            finish();  // close activity if critical failure
        }
        setContentView(R.layout.activity_camera);

        faceOverlayView = findViewById(R.id.faceOverlay);
        previewView = findViewById(R.id.previewView);
        cameraExecutor = Executors.newSingleThreadExecutor();

        // Configure face detector options (fast mode)
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .enableTracking()  // optional, for tracking faces across frames
                        .build();

        faceDetector = FaceDetection.getClient(options);

        startCamera();
    }

    @ExperimentalGetImage
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Preview use case
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // ImageAnalysis use case for ML Kit
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);

                // Front camera selector
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                        .build();

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void analyzeImage(ImageProxy imageProxy) {
        if (imageProxy == null || imageProxy.getImage() == null) {
            if (imageProxy != null) imageProxy.close();
            return;
        }

        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
        int imageWidth = imageProxy.getWidth();
        int imageHeight = imageProxy.getHeight();

        // Let the overlay view know the image source details so it can transform coordinates correctly
        faceOverlayView.setImageSourceInfo(imageWidth, imageHeight, rotationDegrees);

        InputImage inputImage = InputImage.fromMediaImage(
                imageProxy.getImage(),
                rotationDegrees
        );

        faceDetector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    faceOverlayView.setFaces(faces);

                    if (!faces.isEmpty()) {
                        // Convert ImageProxy to Bitmap once
                        Bitmap frameBitmap = imageProxyToBitmap(imageProxy);

                        for (Face face : faces) {
                            // Crop the face Bitmap using bounding box
                            Bitmap faceBitmap = cropToBBox(frameBitmap, face.getBoundingBox());

                            if (faceBitmap != null) {
                                boolean centered = isFaceCentered(face, imageWidth, imageHeight);
                                boolean sizeOk = isFaceSizeOk(face, 150, 400);
                                boolean facingForward = isFacingForward(face);
                                boolean blurryOk = !isBlurry(faceBitmap, 1000);
                                boolean lightingOk = isLightingOk(faceBitmap);

                                // Default box color
                                int boxColor = Color.GREEN;

                                if (!centered) {
                                    boxColor = Color.RED;
                                    showFeedback("⬅️➡️ Move horizontally or ⬆️⬇️ vertically to center your face");
                                }

                                if (!sizeOk) {
                                    boxColor = Color.RED;
                                    showFeedback("↔️ Adjust distance: move closer or back");
                                }

                                if (!facingForward) {
                                    boxColor = Color.RED;
                                    showFeedback("↪️ Turn your face toward the camera");
                                }

                                if (!blurryOk) {
                                    boxColor = Color.RED;
                                    showFeedback("💧 Image is blurry, hold still or adjust lighting");
                                }

                                if (!lightingOk) {
                                    boxColor = Color.RED;
                                    showFeedback("💡 Adjust lighting: too dark or too bright");
                                }

                                // Set the box color based on the checks
                                faceOverlayView.setBoxColor(boxColor);

                                // If all checks passed, capture the embedding
                                if (centered && sizeOk && facingForward && blurryOk && lightingOk) {
                                    float[] embedding = faceEmbedder.getFaceEmbedding(faceBitmap);
                                    Log.d("FaceEmbedding", Arrays.toString(embedding));
                                    showFeedback("✅ Perfect! Face captured.");
                                }
                            }

                        }

                        showFeedback("Faces detected: " + faces.size());
                    } else {
                        faceOverlayView.clearFaces();
                    }
                })
                .addOnFailureListener(e -> Log.e("FaceDetection", "Detection failed", e))
                .addOnCompleteListener(task -> imageProxy.close());
    }

    // Convert ImageProxy to Bitmap
    private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        if (planes.length <= 0) return null;

        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21,
                imageProxy.getWidth(), imageProxy.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, imageProxy.getWidth(), imageProxy.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    // Crop Bitmap to bounding box rectangle (with bounds checking)
    private Bitmap cropToBBox(Bitmap source, Rect bbox) {
        if (source == null || bbox == null) return null;

        int x = Math.max(bbox.left, 0);
        int y = Math.max(bbox.top, 0);
        int width = Math.min(bbox.width(), source.getWidth() - x);
        int height = Math.min(bbox.height(), source.getHeight() - y);

        return Bitmap.createBitmap(source, x, y, width, height);
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        faceDetector.close();
    }

    private boolean isFaceCentered(Face face, int frameWidth, int frameHeight) {
        Rect box = face.getBoundingBox();

        int faceCenterX = box.centerX();
        int faceCenterY = box.centerY();

        int frameCenterX = frameWidth / 2;
        int frameCenterY = frameHeight / 2;

        // Use a tolerance proportional to face width/height
        int tolX = (int)(box.width() * 0.5);   // 50% of face width
        int tolY = (int)(box.height() * 0.5);  // 50% of face height

        boolean horizontalOk = Math.abs(faceCenterX - frameCenterX) <= tolX;
        boolean verticalOk = Math.abs(faceCenterY - frameCenterY) <= tolY;

        return horizontalOk && verticalOk;
    }


    private boolean isFaceSizeOk(Face face, int minSize, int maxSize) {
        int size = Math.max(face.getBoundingBox().width(), face.getBoundingBox().height());
        return size >= minSize && size <= maxSize;
    }

    private boolean isFacingForward(Face face) {
        Float rotY = face.getHeadEulerAngleY(); // left/right
        Float rotZ = face.getHeadEulerAngleZ(); // tilt
        return Math.abs(rotY) < 10 && Math.abs(rotZ) < 10;
    }
    private boolean isLightingOk(Bitmap bitmap) {
        long sum = 0;
        int[] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int pixel : pixels) {
            int r = Color.red(pixel);
            int g = Color.green(pixel);
            int b = Color.blue(pixel);
            int brightness = (r + g + b) / 3;
            sum += brightness;
        }

        double avgBrightness = sum / (double) pixels.length;
        return avgBrightness > 80 && avgBrightness < 200; // adjustable thresholds
    }

    private boolean isBlurry(Bitmap bitmap, double threshold) {
        if (bitmap == null) return true;

        // Downscale for speed
        int newWidth = 100;
        int newHeight = (int) (bitmap.getHeight() * (100.0 / bitmap.getWidth()));
        Bitmap small = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);

        int width = small.getWidth();
        int height = small.getHeight();
        int[] pixels = new int[width * height];
        small.getPixels(pixels, 0, width, 0, 0, width, height);

        // Convert to grayscale
        double[] gray = new double[width * height];
        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            int r = (color >> 16) & 0xff;
            int g = (color >> 8) & 0xff;
            int b = color & 0xff;
            gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }

        // Compute simple Laplacian variance
        double variance = 0;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int i = y * width + x;
                double laplacian = gray[i] - (gray[i - 1] + gray[i + 1] + gray[i - width] + gray[i + width]) / 4.0;
                variance += laplacian * laplacian;
            }
        }

        variance /= (width - 2) * (height - 2);

        //Log.d("BlurCheck", "Variance: " + variance);

        return variance < threshold; // use threshold ~1–5 for front camera
    }

    private void showFeedback(String message) {
        long now = System.currentTimeMillis();
        if (!message.equals(lastFeedback) || now - lastToastTime > 1000) { // 1 sec throttle
            lastFeedback = message;
            lastToastTime = now;
            runOnUiThread(() -> Toast.makeText(this, message, Toast.LENGTH_SHORT).show());
        }
    }

}
