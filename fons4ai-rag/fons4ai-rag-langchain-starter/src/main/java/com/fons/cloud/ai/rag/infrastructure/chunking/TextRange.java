package com.fons.cloud.ai.rag.infrastructure.chunking;

import java.util.Objects;

/**
 * 输入文档中的一个半开字符范围。
 *
 * <p>范围使用 Java {@link String} 的 UTF-16 下标：{@code startInclusive} 包含，
 * {@code endExclusive} 不包含。</p>
 *
 * @author hongqy
 */
public final class TextRange {

    /** 当前分块器输入文本的 UTF-16 起点，包含起点，从 0 开始；不直接代表原文件坐标。 */
    private final int startInclusive;
    /** 当前分块器输入文本的 UTF-16 终点，不包含终点；分组分块时由适配器换算到全文投影。 */
    private final int endExclusive;

    /**
     * 校验范围有效性。
     */
    public TextRange(int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive <= startInclusive) {
            throw new IllegalArgumentException("字符范围必须为非空的有效半开区间");
        }
        this.startInclusive = startInclusive;
        this.endExclusive = endExclusive;
    }

    public int startInclusive() {
        return startInclusive;
    }

    public int endExclusive() {
        return endExclusive;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof TextRange range)) {
            return false;
        }
        return startInclusive == range.startInclusive
                && endExclusive == range.endExclusive;
    }

    @Override
    public int hashCode() {
        return Objects.hash(startInclusive, endExclusive);
    }

    @Override
    public String toString() {
        return "TextRange[" + startInclusive + ", " + endExclusive + ")";
    }
}
