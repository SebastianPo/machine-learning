package group.posacki.application_emotions_detection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class MainActivity extends AppCompatActivity {
    ImageView imageView;
    TextView textView;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/emotions_detector.pb";
    private static final String INPUT_NODE = "reshape_1_input";
    private static final String OUTPUT_NODE = "dense_2/Softmax";
    private TensorFlowInferenceInterface inferenceInterface;
    int imageIDsIndex = 40;
    int[] imagesIDs = {
            R.drawable.happy0,
            R.drawable.happy1,
            R.drawable.happy2,
            R.drawable.happy3,
            R.drawable.happy4,
            R.drawable.happy5,
            R.drawable.happy6,
            R.drawable.happy7,
            R.drawable.happy8,
            R.drawable.happy9,
            R.drawable.happy10,
            R.drawable.happy11,
            R.drawable.happy12,
            R.drawable.happy13,
            R.drawable.happy14,
            R.drawable.happy15,
            R.drawable.happy15,
            R.drawable.happy16,
            R.drawable.happy17,
            R.drawable.happy18,
            R.drawable.happy19,
            R.drawable.happy20,
            R.drawable.sad0,
            R.drawable.sad1,
            R.drawable.sad2,
            R.drawable.sad3,
            R.drawable.sad4,
            R.drawable.sad5,
            R.drawable.sad6,
            R.drawable.sad7,
            R.drawable.sad8,
            R.drawable.sad9,
            R.drawable.sad10,
            R.drawable.sad11,
            R.drawable.sad12,
            R.drawable.sad13,
            R.drawable.sad14,
            R.drawable.sad15,
            R.drawable.sad16,
            R.drawable.sad17,
            R.drawable.sad18,
            R.drawable.sad19,
            R.drawable.sad20,


    };
    Bitmap displayImageBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.results_text_view);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    public void loadImageAction(View view) {

        imageIDsIndex = (imageIDsIndex >= 41) ? 0 : imageIDsIndex + 1;
        Bitmap imageBitmap = BitmapFactory.decodeResource(getResources(), imagesIDs[imageIDsIndex]);
        displayImageBitmap = Bitmap.createScaledBitmap(imageBitmap, 32, 32, true);
        imageView.setImageBitmap(displayImageBitmap);

    }

    public void guessImageAction(View view) {
        float[] pixelBuffer = convertImageToFloatArray();
        float[] results = performInference(pixelBuffer);
        displayResults(results);


    }

    private float[] convertImageToFloatArray() {
        int[] intArray = new int[1024];
        Bitmap outputImageBitmap = BitmapFactory.decodeResource(getResources(), imagesIDs[imageIDsIndex]);
        outputImageBitmap = Bitmap.createScaledBitmap(outputImageBitmap, 32, 32, true);
        outputImageBitmap.getPixels(intArray, 0, 32, 0, 0, 32, 32);
        float[] floatArray = new float[3072];
        for (int i = 0; i < 1024; i++) {
            floatArray[i] = ((intArray[i] >> 16) & 0xff) / 255.0f;
            floatArray[i + 1] = ((intArray[i] >> 8) & 0xff) / 255.0f;
            floatArray[i + 2] = (intArray[i] & 0xff) / 255.0f;
        }
        return floatArray;
    }

    private float[] performInference(float[] pixelBuffer) {
        inferenceInterface.feed(INPUT_NODE, pixelBuffer, 1, 32, 32, 3);
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        float[] results = new float[2];
        inferenceInterface.fetch(OUTPUT_NODE, results);
        return results;

    }

    private void displayResults(float[] results) {
        if (results[0] > results[1]) {
            textView.setText("Model predicts: Happy");
        } else if (results[0] < results[1]) {
            textView.setText("Model predicts: Sad");
        } else {
            textView.setText("Model predicts: Neither");
        }
    }
}

