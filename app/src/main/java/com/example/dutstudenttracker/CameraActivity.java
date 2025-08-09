package com.example.dutstudenttracker;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

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

                            if (faceEmbedder != null && faceBitmap != null) {
                                float[] embedding = faceEmbedder.getFaceEmbedding(faceBitmap);

                                // TODO: Use the embedding - e.g., send to server or compare with stored embeddings
                                Log.d("FaceEmbedding", "Embedding length: " + embedding.length);
                            }
                        }

                        runOnUiThread(() ->
                                Toast.makeText(this, "Faces detected: " + faces.size(), Toast.LENGTH_SHORT).show()
                        );
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
}
