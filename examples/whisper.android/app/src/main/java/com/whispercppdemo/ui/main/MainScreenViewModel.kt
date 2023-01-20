package com.whispercppdemo.ui.main

import android.app.Application
import android.content.ActivityNotFoundException
import android.content.Intent
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat.startActivity
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.whispercppdemo.recorder.Recorder
import com.whispercppdemo.recorder.ThoughtProcessor
import com.whispercppdemo.whisper.WhisperContext
import kotlinx.coroutines.*
import java.io.File


private const val LOG_TAG = "MainScreenViewModel"

class MainScreenViewModel(private val application: Application) : ViewModel() {
    var canTranscribe by mutableStateOf(false)
        private set
    var dataLog by mutableStateOf("")
        private set
    var debugLog by mutableStateOf("")
        private set
    var vadDetection by mutableStateOf("")
        private set
    var isRecording by mutableStateOf(false)
        private set

    private val modelsPath = File(application.filesDir, "models")
    private val samplesPath = File(application.filesDir, "samples")
    private var recorder: Recorder = Recorder()
    private var whisperContext: WhisperContext? = null
    private var sileroModel: ByteArray
    private val processor: ThoughtProcessor
    private lateinit var tts: TextToSpeech

    init {
        viewModelScope.launch {
            loadData()
        }

        val inputStream = application.assets.open("vad/silero_vad.onnx")
        sileroModel = inputStream.readBytes()
        processor = ThoughtProcessor(application.applicationContext)
    }

    private suspend fun loadData() {
        printMessage("Loading data...\n")
        try {
            loadBaseModel()
            canTranscribe = true
        } catch (e: Exception) {
            Log.w(LOG_TAG, e)
            printMessage("${e.localizedMessage}\n")
        }
    }

    private suspend fun printMessage(msg: String) = withContext(Dispatchers.Main) {
        dataLog += msg
    }

    private suspend fun loadBaseModel() = withContext(Dispatchers.IO) {
        printMessage("Loading model...\n")
        val models = application.assets.list("models/")
        if (models != null) {
            val inputStream = application.assets.open("models/" + models[0])
            whisperContext = WhisperContext.createContextFromInputStream(inputStream)
            printMessage("Loaded model ${models[0]}.\n")
        }
    }

    fun toggleRecord() = viewModelScope.launch {
        try {
            if (isRecording) {
                recorder.stopRecording()
                isRecording = false
            } else {
                val onDebug: (String) -> Unit = { s ->
                    debugLog = s
                }

                val onVad: (Boolean) -> Unit = { detection ->
                    vadDetection = detection.toString()
                }

                val onData: (FloatArray, () -> Any?) -> Unit = { data, cb ->
                    class l : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                        }

                        override fun onDone(utteranceId: String?) {
                            cb()
                        }

                        @Deprecated("Deprecated in Java")
                        override fun onError(utteranceId: String?) {
                            cb()
                        }

                    }
                    runBlocking {
                        var callCallback = true
                        try {

                            printMessage("you: ")
                            val text = whisperContext?.transcribeData(data)
                            val processedText = processor.preProcess(text)
                            printMessage("$processedText\n")

                            if (processedText.isBlank()) {
                                return@runBlocking
                            }
                            var genResult = processor.process(processedText)
                            var exitLoop = false

                            while (!exitLoop) {
                                exitLoop = true
                                for (res in genResult) {
                                    printMessage("${res.to}: ")
                                    printMessage("${res.q}\n")

                                    if (res.to == "human") {
                                        if (tts != null) {
                                            tts.speak(res.q, TextToSpeech.QUEUE_FLUSH, null, "")
                                            tts.setOnUtteranceProgressListener(l())
                                            callCallback = false
                                        }
                                    } else if (res.to == "gps") {
                                        printMessage("Finding ${res.q}\n")
                                        val places = processor.findPlaces(res.q)
                                        printMessage("Found ${places}\n")

                                        genResult = processor.process(places)
                                        exitLoop = false
                                        callCallback = false
                                    } else if (res.to == "navigate") {
                                        //val lat = res.lat
                                        //var lng = res.lng
                                        printMessage("Coord: ${res.q}\n")
                                        if (res.q.isNotBlank()) {
                                            processor.openWaze(res.q)
                                        }
                                    }
                                }
                            }

                        } catch (e: Exception) {
                            Log.w(LOG_TAG, e)
                            printMessage("${e.localizedMessage}\n")
                            callCallback = true
                        } finally {
                            if (callCallback) {
                                cb()
                            }
                        }
                    }
                }

                recorder.startRecording(sileroModel, onData, onVad, onDebug) { e ->
                    viewModelScope.launch {
                        withContext(Dispatchers.Main) {
                            printMessage("${e.localizedMessage}\n")
                            isRecording = false
                        }
                    }
                }

                isRecording = true
            }
        } catch (e: Exception) {
            Log.w(LOG_TAG, e)
            printMessage("${e.localizedMessage}\n")
            isRecording = false
        }
    }

    override fun onCleared() {
        runBlocking {
            whisperContext?.release()
            whisperContext = null
        }
    }

    fun setTTS(tts: TextToSpeech) {
        this.tts = tts
    }

    companion object {
        fun factory() = viewModelFactory {
            initializer {
                val application =
                    this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as Application
                MainScreenViewModel(application)
            }
        }
    }
}
