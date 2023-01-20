package com.whispercppdemo.recorder

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer

class SileroVad(
    private var env: OrtEnvironment,
    model: ByteArray,
    private var sampleRate: Int,
    frameSize: Int,
    private var threshold: Float,
    minSilenceDurationMs: Int,
    speechPadMs: Int
) {
    private val session: OrtSession
    private val windowSampleSize: Int
    private val minSilenceSamples: Int
    private val speechPadSamples: Int
    private val sizeHC = 2 * 1 * 64 // Fixed
    private val hData: MutableList<Float>
    private val cData: MutableList<Float>

    // Model states
    private var triggerd = false
    private var speechStart: Int = 0
    private var speechEnd: Int = 0
    private var tempEnd: Int = 0
    private var currentSample: Int = 0
    //private var output: Float = 0.0f

    init {
        val o = OrtSession.SessionOptions()
        o.setInterOpNumThreads(1)
        o.setIntraOpNumThreads(1)
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

        session = env.createSession(model, o)

        val srPerMs = sampleRate / 1000
        val windowSampleSize = frameSize * srPerMs
        this.minSilenceSamples = srPerMs * minSilenceDurationMs
        this.speechPadSamples = srPerMs * speechPadMs
        this.windowSampleSize = windowSampleSize

        hData = MutableList<Float>(sizeHC) { _ -> 0.0f }
        cData = MutableList<Float>(sizeHC) { _ -> 0.0f }
    }

    fun reset() {
        for (i in 0 until hData.size) {
            hData[i] = 0.0f
        }
        for (i in 0 until cData.size) {
            cData[i] = 0.0f
        }

        triggerd = false;
        tempEnd = 0;
        currentSample = 0;
    }

    fun predict(data: FloatArray): Boolean {
        val dataBuf = FloatBuffer.wrap(data)
        val srDataBuf = LongBuffer.wrap(longArrayOf(sampleRate.toLong()))
        val hDataBuf = FloatBuffer.wrap(hData.toFloatArray())
        val cDataBuf = FloatBuffer.wrap(cData.toFloatArray())

        val input = OnnxTensor.createTensor(env, dataBuf, longArrayOf(1, windowSampleSize.toLong()))
        val sr = OnnxTensor.createTensor(env, srDataBuf, longArrayOf(1))
        val h = OnnxTensor.createTensor(env, hDataBuf, longArrayOf(2, 1, 64))
        val c = OnnxTensor.createTensor(env, cDataBuf, longArrayOf(2, 1, 64))

        val inputNodes = mapOf(
            "input" to input,
            "sr" to sr,
            "h" to h,
            "c" to c)

        val outputNodes = setOf<String>(
            "output",
            "hn",
            "cn"
        )

        val outputs = session.run(inputNodes, outputNodes)

        val output = ((outputs.get(0).value) as Array<FloatArray>)[0][0]
        val hn = ((outputs.get(1).value) as Array<Array<FloatArray>>)
        val cn = ((outputs.get(2).value) as Array<Array<FloatArray>>)

        val hnf = hn.flatten().flatMap { it.asIterable() }
        val cnf = cn.flatten().flatMap { it.asIterable() }

        // Output probability & update h,c recursively
        for (i in 0 until hnf.size) {
            hData[i] = hnf[i]
        }
        for (i in 0 until cnf.size) {
            cData[i] = cnf[i]
        }

        println("output: ${output}")

        // Push forward sample index
        currentSample += windowSampleSize

        // Reset temp_end when > threshold
        if ((output >= threshold) && (tempEnd != 0))
        {
            tempEnd = 0;
        }

        // 1) Silence
        if ((output < threshold) && (triggerd == false))
        {
            // printf("{ silence: %.3f s }\n", 1.0 * current_sample / sample_rate);
        }
        // 2) Speaking
        if ((output >= (threshold - 0.15)) && (triggerd == true))
        {
            // printf("{ speaking_2: %.3f s }\n", 1.0 * current_sample / sample_rate);
        }

        // 3) Start
        if ((output >= threshold) && (triggerd == false))
        {
            triggerd = true;

            // minus window_size_samples to get precise start time point.
            speechStart = currentSample - windowSampleSize - speechPadSamples;
            println("{ start: ${1.0 * speechStart / sampleRate} s }");
        }

        // 4) End
        if ((output < (threshold - 0.15)) && (triggerd == true))
        {
            if (tempEnd == 0)
            {
                tempEnd = currentSample;
            }
            // a. silence < min_slience_samples, continue speaking
            if ((currentSample - tempEnd) < minSilenceSamples)
            {
                // printf("{ speaking_4: %.3f s }\n", 1.0 * current_sample / sample_rate);
                // printf("");
            }
            // b. silence >= min_slience_samples, end speaking
            else
            {
                speechEnd = currentSample + speechPadSamples;
                tempEnd = 0;
                triggerd = false;
                println("{ end: ${1.0 * speechEnd / sampleRate} s }");
            }
        }

        return triggerd
    }
}