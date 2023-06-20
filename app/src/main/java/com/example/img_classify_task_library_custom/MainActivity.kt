package com.example.img_classify_task_library_custom

import android.graphics.*
import android.graphics.ImageDecoder.ImageInfo
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class MainActivity : AppCompatActivity() {

    private lateinit var loadImageButton: Button
    private lateinit var imageView: ImageView
    private lateinit var selectedImage: Bitmap
    private lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.image_view)
        resultTextView = findViewById(R.id.result_textview)
        loadImageButton = findViewById(R.id.load_image_button)

        loadImageButton.setOnClickListener {
            pickImage.launch("image/*")
        }

        imageView.setOnClickListener {
            runTensorFlowLiteImgClassify(selectedImage)
        }

    }
    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                "To classify, click the image".also { resultTextView.text = it } // Clear text
                val source = ImageDecoder.createSource(contentResolver, uri)
                selectedImage = ImageDecoder.decodeBitmap(source)
                //ImageDecoder.decodeBitmap(source)
                { imageDecoder: ImageDecoder, imageInfo: ImageInfo?, source1: ImageDecoder.Source? ->
                    imageDecoder.isMutableRequired = true
                }
                imageView.setImageBitmap(selectedImage)
            }
        }
    // Task Library Object Detection function
    private fun runTensorFlowLiteImgClassify(bitmap: Bitmap) {

        // Initialization
        val options = ImageClassifier.ImageClassifierOptions.builder()
            .setBaseOptions(BaseOptions.builder().build())   //.useGpu()    before .build()
            .setMaxResults(3)
            .build()
        val imageClassifier = ImageClassifier.createFromFileAndOptions(
            //this, "horses_humans_model_and_labels.tflite", options
            this, "lite-model_imagenet_mobilenet_v3_small_100_224_classification_5_metadata_1.tflite", options
        )

        // Run inference
        val inputImage = TensorImage.fromBitmap(bitmap)
        val outputs = imageClassifier.classify(inputImage)


        // Show the classification results in a Toast
        val topResult = outputs.firstOrNull()
        val label = topResult?.categories?.get(0)?.label
        val score = topResult?.categories?.get(0)?.score
        val confidence = score?.times(1)
        val message = "Label: $label\nConfidence: $confidence%"
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()


    }
}



