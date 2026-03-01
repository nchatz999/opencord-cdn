/**
 * BSConformer AudioWorklet Processor
 *
 * Loads the standalone WASM binary directly on the audio thread,
 * buffers 128-sample render quanta into 480-sample denoiser frames,
 * and drains processed output back at the render quantum rate.
 */

const FRAME = 480; // bsconf_process_frame expects 480 samples

class BSConformerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ready = false;
    this.bypass = false;

    // Input accumulation buffer (FRAME + 128 worst case)
    this.accum = new Float32Array(FRAME + 128);
    this.accumLen = 0;

    // Output drain queue
    this.outQueue = [];
    this.outOffset = 0;

    this.port.onmessage = (e) => this.onMessage(e);
  }

  onMessage(e) {
    if (e.data.type === "init") {
      try {
        this.initWasm(e.data.wasmModule);
        this.ready = true;
        this.port.postMessage({ type: "ready" });
      } catch (err) {
        this.port.postMessage({ type: "error", message: err.toString() });
      }
    } else if (e.data.type === "bypass") {
      this.bypass = e.data.value;
    }
  }

  initWasm(wasmModule) {
    // Build import object by inspecting what the WASM binary expects.
    // STANDALONE_WASM uses wasi_snapshot_preview1 imports -- all can be
    // no-op stubs since the C code does no I/O in the processing path.
    const needed = WebAssembly.Module.imports(wasmModule);
    const importObj = {};
    for (const imp of needed) {
      if (!importObj[imp.module]) importObj[imp.module] = {};
      if (imp.kind === "function") {
        importObj[imp.module][imp.name] = () => 0;
      }
    }
    // proc_exit should throw so we notice if it fires
    if (importObj.wasi_snapshot_preview1) {
      importObj.wasi_snapshot_preview1.proc_exit = (code) => {
        throw new Error("bsconf proc_exit: " + code);
      };
    }

    // Synchronous instantiation -- avoids async/await issues in AudioWorklet
    this.port.postMessage({ type: "log", message: "Instantiating WASM..." });
    const instance = new WebAssembly.Instance(wasmModule, importObj);
    const exp = instance.exports;
    this.wasm = exp;
    this.memory = exp.memory;

    // bsconf_create computes 4 DFT matrices (~1.8M trig calls) -- takes a moment
    this.port.postMessage({ type: "log", message: "Creating model (computing DFT matrices)..." });
    this.modelPtr = exp.bsconf_create();
    if (!this.modelPtr) throw new Error("bsconf_create returned null");

    this.port.postMessage({ type: "log", message: "Loading weights..." });
    exp.bsconf_load_weights(this.modelPtr);
    this.statePtr = exp.bsconf_state_create();
    if (!this.statePtr) throw new Error("bsconf_state_create returned null");

    // Allocate I/O buffers in WASM heap (480 float32 = 1920 bytes each)
    this.inputPtr = exp.malloc(FRAME * 4);
    this.outputPtr = exp.malloc(FRAME * 4);
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0][0];

    if (!input || !this.ready) {
      output.fill(0);
      return true;
    }

    if (this.bypass) {
      output.set(input);
      return true;
    }

    // --- Accumulate input ---
    this.accum.set(input, this.accumLen);
    this.accumLen += input.length;

    // --- Process complete frames ---
    if (this.accumLen >= FRAME) {
      const wasmIn = new Float32Array(
        this.memory.buffer,
        this.inputPtr,
        FRAME
      );
      const wasmOut = new Float32Array(
        this.memory.buffer,
        this.outputPtr,
        FRAME
      );

      wasmIn.set(this.accum.subarray(0, FRAME));
      this.wasm.bsconf_process_frame(
        this.modelPtr,
        this.statePtr,
        this.inputPtr,
        this.outputPtr
      );

      // Copy output out of WASM memory (buffer may be reused next frame)
      this.outQueue.push(new Float32Array(wasmOut));

      // Shift remainder
      const rem = this.accumLen - FRAME;
      if (rem > 0) {
        this.accum.copyWithin(0, FRAME, FRAME + rem);
      }
      this.accumLen = rem;
    }

    // --- Drain output ---
    let written = 0;
    while (written < output.length && this.outQueue.length > 0) {
      const frame = this.outQueue[0];
      const avail = frame.length - this.outOffset;
      const take = Math.min(avail, output.length - written);
      output.set(frame.subarray(this.outOffset, this.outOffset + take), written);
      written += take;
      this.outOffset += take;
      if (this.outOffset >= frame.length) {
        this.outQueue.shift();
        this.outOffset = 0;
      }
    }
    // Silence for any remaining samples (startup latency)
    if (written < output.length) {
      output.fill(0, written);
    }

    return true;
  }
}

registerProcessor("bsconformer-processor", BSConformerProcessor);
