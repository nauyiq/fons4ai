package com.fons.cloud.ai.agent.observability.core;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.observability.api.TraceSink;
import com.fons.cloud.ai.agent.observability.model.TraceRecord;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Pattern;

/**
 * 将每个 Agent Trace 按 JSON Lines 格式写入独立文件的输出端。
 *
 * @author hongqy
 */
public final class JsonlTraceSink implements TraceSink {

    /** 可直接作为安全文件名使用的 Run 标识格式。 */
    private static final Pattern SAFE_FILE_NAME = Pattern.compile("[A-Za-z0-9._-]{1,128}");

    /** 用于限制同一文件并发写入的固定锁数量。 */
    private static final int WRITE_LOCK_COUNT = 64;

    /** JSONL 文件保存目录的绝对规范路径。 */
    private final Path storageDirectory;

    /** 基于文件路径散列选择的写入锁，避免同一 Trace 的内容交叉。 */
    private final ReentrantLock[] writeLocks = createWriteLocks();

    /**
     * 创建 JSONL 输出端。
     *
     * @param storageDirectory JSONL 文件保存目录
     */
    public JsonlTraceSink(Path storageDirectory) {
        this.storageDirectory = Objects.requireNonNull(storageDirectory, "storageDirectory cannot be null")
                .toAbsolutePath()
                .normalize();
    }

    @Override
    public void append(TraceRecord record) throws Exception {
        Objects.requireNonNull(record, "record cannot be null");
        Path traceFile = storageDirectory.resolve(toFileName(record.context().runId()));
        ReentrantLock writeLock = writeLock(traceFile);
        writeLock.lock();
        try {
            Files.createDirectories(storageDirectory);
            Files.writeString(
                    traceFile,
                    JSON.toJSONString(record) + "\n",
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } finally {
            writeLock.unlock();
        }
    }

    /**
     * 创建固定数量的文件写入锁。
     *
     * @return 文件写入锁数组
     */
    private static ReentrantLock[] createWriteLocks() {
        ReentrantLock[] locks = new ReentrantLock[WRITE_LOCK_COUNT];
        for (int index = 0; index < locks.length; index++) {
            locks[index] = new ReentrantLock();
        }
        return locks;
    }

    /**
     * 根据文件路径选择稳定的写入锁。
     *
     * @param traceFile Trace 文件路径
     * @return 对应写入锁
     */
    private ReentrantLock writeLock(Path traceFile) {
        int index = Math.floorMod(traceFile.hashCode(), writeLocks.length);
        return writeLocks[index];
    }

    /**
     * 将 Run 标识转换为不可越过保存目录的文件名。
     *
     * @param runId Run 唯一标识
     * @return JSONL 文件名
     */
    private static String toFileName(String runId) {
        if (SAFE_FILE_NAME.matcher(runId).matches()) {
            return runId + ".jsonl";
        }
        return UUID.nameUUIDFromBytes(runId.getBytes(StandardCharsets.UTF_8)) + ".jsonl";
    }
}
