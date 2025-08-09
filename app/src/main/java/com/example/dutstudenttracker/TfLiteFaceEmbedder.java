package com.example.dutstudenttracker;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class TfLiteFaceEmbedder {

    private static final int INPUT_SIZE = 160;    // FaceNet input size 160x160
    private static final int EMBEDDING_SIZE = 128; // FaceNet output size

    private Interpreter tflite;

    public TfLiteFaceEmbedder(Context context, String modelFileName) throws IOException {
        tflite = new Interpreter(loadModelFile(context, modelFileName));
    }

    // Load TFLite model from assets
    private MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Preprocess Bitmap -> ByteBuffer to feed into model
    private ByteBuffer preprocessBitmap(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        resizedBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        // Normalize pixels to [-1, 1] as FaceNet expects
        for (int pixelValue : intValues) {
            float r = ((pixelValue >> 16) & 0xFF);
            float g = ((pixelValue >> 8) & 0xFF);
            float b = (pixelValue & 0xFF);

            imgData.putFloat((r - 127.5f) / 128.0f);
            imgData.putFloat((g - 127.5f) / 128.0f);
            imgData.putFloat((b - 127.5f) / 128.0f);
        }
        return imgData;
    }

    // Run inference on a face Bitmap and get the embedding vector
    public float[] getFaceEmbedding(Bitmap faceBitmap) {
        ByteBuffer input = preprocessBitmap(faceBitmap);

        float[][] output = new float[1][EMBEDDING_SIZE];
        tflite.run(input, output);

        return l2Normalize(output[0]);
    }

    // L2 normalize the embedding for consistency
    private float[] l2Normalize(float[] embedding) {
        double sum = 0;
        for (float val : embedding) {
            sum += val * val;
        }
        double norm = Math.sqrt(sum);
        float[] normalized = new float[embedding.length];
        for (int i = 0; i < embedding.length; i++) {
            normalized[i] = (float)(embedding[i] / (norm + 1e-10));
        }
        return normalized;
    }

    // Close interpreter resources when done
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }
}
