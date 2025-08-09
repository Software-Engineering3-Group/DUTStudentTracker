package com.example.dutstudenttracker;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import com.google.mlkit.vision.face.Face;

import java.util.ArrayList;
import java.util.List;

public class FaceOverlayView extends View {

    private List<Face> faces = new ArrayList<>();
    private Paint boxPaint;
    private int imageWidth;
    private int imageHeight;
    private boolean isFrontFacing;
    private int rotationDegrees;

    public FaceOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);

        boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);
        boxPaint.setAntiAlias(true);
    }

    public void setFaces(List<Face> faces) {
        this.faces = faces;
        invalidate(); // Redraw the view
    }

    public void clearFaces() {
        this.faces = new ArrayList<>();
        invalidate();
    }

    public void setImageSourceInfo(int imageWidth, int imageHeight, int rotationDegrees) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.rotationDegrees = rotationDegrees;

        // You can set this manually if you know you're using the front cam
        this.isFrontFacing = true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (faces == null || imageWidth == 0 || imageHeight == 0) return;

        for (Face face : faces) {
            Rect boundingBox = face.getBoundingBox();

            // Map face bounds to view coordinates
            RectF mappedBox = translateRect(boundingBox);

            // Draw the face rectangle
            canvas.drawRect(mappedBox, boxPaint);
        }
    }

    private RectF translateRect(Rect boundingBox) {
        float viewWidth = getWidth();
        float viewHeight = getHeight();

        float scaleX;
        float scaleY;

        // Consider image rotation when calculating scale
        if (rotationDegrees == 0 || rotationDegrees == 180) {
            scaleX = viewWidth / (float) imageWidth;
            scaleY = viewHeight / (float) imageHeight;
        } else {
            // Swap width/height for portrait images
            scaleX = viewWidth / (float) imageHeight;
            scaleY = viewHeight / (float) imageWidth;
        }

        float left = boundingBox.left * scaleX;
        float top = boundingBox.top * scaleY;
        float right = boundingBox.right * scaleX;
        float bottom = boundingBox.bottom * scaleY;

        RectF scaledBox = new RectF(left, top, right, bottom);

        // Mirror for front camera
        if (isFrontFacing) {
            float mirroredLeft = viewWidth - scaledBox.right;
            float mirroredRight = viewWidth - scaledBox.left;
            scaledBox.left = mirroredLeft;
            scaledBox.right = mirroredRight;
        }

        return scaledBox;
    }
}
