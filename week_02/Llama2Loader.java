import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class Llama2Loader {
    // === Structs mirroring llama2.c ===
    public static final class Config {
        public int dim;
        public int hiddenDim;
        public int nLayers;
        public int nHeads;
        public int nKvHeads;
        public int vocabSize;
        public int seqLen;
        public boolean sharedWeights;

        @Override public String toString() {
            return "Config{" +
                    "dim=" + dim +
                    ", hiddenDim=" + hiddenDim +
                    ", nLayers=" + nLayers +
                    ", nHeads=" + nHeads +
                    ", nKvHeads=" + nKvHeads +
                    ", vocabSize=" + vocabSize +
                    ", seqLen=" + seqLen +
                    ", sharedWeights=" + sharedWeights +
                    '}';
        }
    }

    public static final class TransformerWeights {
        public float[] data;
        // each tensor offset (index into data[])
        public int tokenEmbeddingTable;
        public int rmsAttWeight;
        public int wq, wk, wv, wo;
        public int rmsFfnWeight;
        public int w1, w2, w3;
        public int rmsFinalWeight;
        // RoPE placeholders
        public int ropeFreqReal;
        public int ropeFreqImag;
        public int wcls;
    }

    public static final class Model {
        public Config config;
        public TransformerWeights weights;
        public long fileSizeBytes;
    }

    // small helper to ensure buffer is fully filled
    private static void readFully(FileChannel ch, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) {
            int n = ch.read(buf);
            if (n < 0) throw new EOFException("Unexpected EOF while reading");
        }
    }

    // === Loader ===
    public static Model readCheckpoint(Path checkpointPath) throws IOException {
        try (FileChannel ch = FileChannel.open(checkpointPath, StandardOpenOption.READ)) {
            long fileSize = ch.size();
            if (fileSize < 28) throw new EOFException("File too small to contain Config header (need 28 bytes)");

            // 1) read header (7 little-endian int32)
            ByteBuffer head = ByteBuffer.allocate(28).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, head);
            head.flip();

            Config cfg = new Config();
            cfg.dim       = head.getInt();
            cfg.hiddenDim = head.getInt();
            cfg.nLayers   = head.getInt();
            cfg.nHeads    = head.getInt();
            cfg.nKvHeads  = head.getInt();
            int rawVocab  = head.getInt();
            cfg.seqLen    = head.getInt();
            cfg.sharedWeights = rawVocab > 0;
            cfg.vocabSize     = Math.abs(rawVocab);

            // 2) read the remaining weights as little-endian float[]
            long remaining = fileSize - 28;
            if ((remaining % 4) != 0) {
                throw new IOException("Weights segment is not a multiple of 4 bytes");
            }
            if (remaining > Integer.MAX_VALUE) {
                throw new IOException("Weights too large to fit into a single Java array");
            }

            int nFloats = (int) (remaining / 4);
            ByteBuffer buf = ByteBuffer.allocateDirect((int) remaining).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, buf);
            buf.flip();

            float[] data = new float[nFloats];
            FloatBuffer fb = buf.asFloatBuffer();
            fb.get(data);

            // 3) Map tensor offsets
            TransformerWeights w = new TransformerWeights();
            w.data = data;

            int ptr = 0;
            final int dim = cfg.dim;
            final int headSize = dim / cfg.nHeads;
            final int kvDim = (cfg.dim * cfg.nKvHeads) / cfg.nHeads;
            final long nLayers = Integer.toUnsignedLong(cfg.nLayers);

            // token_embedding_table: (vocab_size, dim)
            w.tokenEmbeddingTable = ptr; ptr += cfg.vocabSize * dim;

            // rms_att_weight: (layer, dim)
            w.rmsAttWeight = ptr; ptr += (int) (nLayers * dim);

            // wq: (layer, dim, n_heads * head_size) -> n_layers * dim * dim
            w.wq = ptr; ptr += (int) (nLayers * dim * dim);

            // wk, wv: (layer, dim, n_kv_heads * head_size) -> n_layers * dim * kv_dim
            w.wk = ptr; ptr += (int) (nLayers * dim * kvDim);
            w.wv = ptr; ptr += (int) (nLayers * dim * kvDim);

            // wo: (layer, n_heads * head_size, dim) -> n_layers * dim * dim
            w.wo = ptr; ptr += (int) (nLayers * dim * dim);

            // rms_ffn_weight: (layer, dim)
            w.rmsFfnWeight = ptr; ptr += (int) (nLayers * dim);

            // w1, w2, w3
            w.w1 = ptr; ptr += (int) (nLayers * dim * cfg.hiddenDim);
            w.w2 = ptr; ptr += (int) (nLayers * cfg.hiddenDim * dim);
            w.w3 = ptr; ptr += (int) (nLayers * dim * cfg.hiddenDim);

            // rms_final_weight: (dim)
            w.rmsFinalWeight = ptr; ptr += dim;

            // skip two RoPE halves
            w.ropeFreqReal = ptr; ptr += cfg.seqLen * headSize / 2;
            w.ropeFreqImag = ptr; ptr += cfg.seqLen * headSize / 2;

            // wcls: shared -> alias embedding; unshared -> extra (vocab_size, dim)
            if (cfg.sharedWeights) {
                w.wcls = w.tokenEmbeddingTable;
            } else {
                w.wcls = ptr; ptr += cfg.vocabSize * dim;
            }

            // 4) sanity: consumed floats == data.length
            if (ptr != data.length) {
                throw new IOException(String.format(
                        "Weight buffer length mismatch: consumed %d floats, file has %d floats",
                        ptr, data.length));
            }

            Model model = new Model();
            model.config = cfg;
            model.weights = w;
            model.fileSizeBytes = fileSize;
            return model;
        }
    }

    // === Tests ===
    private static void assertTrue(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }

    private static void testLoadStories15M(Path path) throws IOException {
        System.out.println("[TEST] Loading: " + path);
        Model m = readCheckpoint(path);
        Config c = m.config;
        TransformerWeights w = m.weights;

        // 1) Basic config sanity
        assertTrue(c.dim > 0 && c.hiddenDim > 0 && c.nLayers > 0 && c.nHeads > 0 && c.nKvHeads > 0 && c.vocabSize > 0 && c.seqLen > 0,
                "Config fields must be positive");
        assertTrue(c.dim % c.nHeads == 0, "dim must be divisible by nHeads (for headSize)");

        // 2) Expected float count matches file size
        int headSize = c.dim / c.nHeads;
        int kvDim = (c.dim * c.nKvHeads) / c.nHeads;
        long nLayers = Integer.toUnsignedLong(c.nLayers);
        long expectedFloats = 0;
        expectedFloats += (long) c.vocabSize * c.dim;             // token_embedding_table
        expectedFloats += nLayers * c.dim;                        // rms_att_weight
        expectedFloats += nLayers * (long) c.dim * c.dim;         // wq
        expectedFloats += nLayers * (long) c.dim * kvDim;         // wk
        expectedFloats += nLayers * (long) c.dim * kvDim;         // wv
        expectedFloats += nLayers * (long) c.dim * c.dim;         // wo
        expectedFloats += nLayers * c.dim;                        // rms_ffn_weight
        expectedFloats += nLayers * (long) c.dim * c.hiddenDim;   // w1
        expectedFloats += nLayers * (long) c.hiddenDim * c.dim;   // w2
        expectedFloats += nLayers * (long) c.dim * c.hiddenDim;   // w3
        expectedFloats += c.dim;                                  // rms_final_weight
        expectedFloats += (long) c.seqLen * headSize;             // rope two halves total
        if (!c.sharedWeights) expectedFloats += (long) c.vocabSize * c.dim; // wcls if unshared

        long actualFloats = (m.fileSizeBytes - 28) / 4; // 7 * int32 header
        assertTrue(expectedFloats == actualFloats,
                "Expected floats (" + expectedFloats + ") != actual (" + actualFloats + ")");

        // 3) wcls tying rule
        if (c.sharedWeights) {
            assertTrue(w.wcls == w.tokenEmbeddingTable, "wcls should alias tokenEmbeddingTable when sharedWeights=true");
        } else {
            assertTrue(w.wcls > w.rmsFinalWeight, "wcls should come after rms_final_weight when unshared");
        }

        System.out.println("[PASS] stories15M.bin loaded. " + c);
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: java Llama2Loader <path-to-checkpoint.bin>\n" +
                    "Example: java Llama2Loader stories15M.bin");
            System.exit(2);
        }
        Path p = Paths.get(args[0]);
        testLoadStories15M(p);
    }
}
