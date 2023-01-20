package com.whispercppdemo.recorder

import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Process
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class Recorder {
    private val scope: CoroutineScope = CoroutineScope(
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    )
    private var recorder: AudioRecordThread? = null

    suspend fun startRecording(
        sileroModel: ByteArray,
        onData: (FloatArray, done: () -> Any?) -> Unit,
        onVad: (Boolean) -> Unit,
        onDebug: (String) -> Unit,
        onError: (Exception) -> Unit
    ) = withContext(scope.coroutineContext) {
        recorder = AudioRecordThread(sileroModel, onError, onData, onVad, onDebug)
        recorder?.start()
    }

    suspend fun stopRecording() = withContext(scope.coroutineContext) {
        recorder?.stopRecording()
        @Suppress("BlockingMethodInNonBlockingContext")
        recorder?.join()
        recorder = null
    }
}

private class AudioRecordThread(
    private var sileroModel: ByteArray,
    private val onError: (Exception) -> Unit,
    private val onData: (FloatArray, done: () -> Any?) -> Unit,
    private val onVad: (Boolean) -> Unit,
    private val onDebug: (String) -> Unit
) :
    Thread("AudioRecorder") {
    private var quit = AtomicBoolean(false)
    private var pause = AtomicBoolean(false)

    @SuppressLint("MissingPermission")
    override fun run() {
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            val env = OrtEnvironment.getEnvironment()

            val sampleRateHz = 16000
            val frameSize = 160

            val svad = SileroVad(
                env,
                sileroModel,
                sampleRateHz,
                frameSize,
                0.6f,
                1500,
                0)

            val bufferSize = AudioRecord.getMinBufferSize(
                sampleRateHz,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )// * 4

            //val buffer = ShortArray(bufferSize / 2)
            val buffer = ShortArray(frameSize * 2)

            println("SampleRate: $sampleRateHz")
            println("bufferSize: $bufferSize")
            println("bufferF: ${buffer.size}")

            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRateHz,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )

            try {
                audioRecord.startRecording()

                var curFrame = 0
                val allData = mutableListOf<Float>()
                val newData = mutableListOf<Float>()
                var lastVad = false

                onVad(lastVad)

                while (!quit.get()) {
                    if (pause.get()) {
                        Thread.sleep(1000)
                        continue
                    }

                    val read = audioRecord.read(buffer, 0, buffer.size)

                    if (read > 0) {
                        for (i in 0 until read) {
                            newData.add((buffer[i] / 32767.0f).coerceIn(-1f..1f))
                        }

                        val srPerMs = 16 //sr / 1000
                        val windowSampleSize = 160 * srPerMs
                        if (newData.size < windowSampleSize) {
                            continue
                        }

                        val lastSpeaking = lastVad
                        val isSpeaking = svad.predict(newData.toFloatArray())
                        if (lastVad != isSpeaking) {
                            lastVad = isSpeaking
                            onVad(isSpeaking)
                        }

                        onDebug("Speech: $isSpeaking (${curFrame++})")

                        if (isSpeaking) {
                            allData.addAll(newData)
                        } else if (lastSpeaking) {
                            allData.addAll(newData)
                            pause.set(true)
                            onData(allData.toFloatArray()) {
                                pause.set(false)
                                allData.clear()
                                svad.reset()
                            }
                            allData.clear()
                            svad.reset()
                        }

                        newData.clear()

                    } else {
                        throw java.lang.RuntimeException("audioRecord.read returned $read")
                    }
                }
            } finally {
                onDebug("Speech: Stopped")
                audioRecord.release()
                env.close()
            }
        } catch (e: Exception) {
            onError(e)
        }
    }

    fun stopRecording() {
        quit.set(true)
    }
}