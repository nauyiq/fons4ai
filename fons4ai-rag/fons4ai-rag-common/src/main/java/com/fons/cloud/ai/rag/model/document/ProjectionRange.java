package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

import java.util.Objects;

/**
 * 解析文档的连续正文投影中的 UTF-16 半开区间。
 *
 * <p>该坐标只对应 {@link DocumentProjection#getText()}，不是原始文件中的字符、页或时间位置。
 * 原文件位置由 {@link SourceLocation} 单独表示。</p>
 *
 * @author hongqy
 */
@Getter
public final class ProjectionRange {

    /**
     * 连续投影中的 UTF-16 起始下标，从零开始，包含该位置。
     */
    private final int startInclusive;
    /**
     * 连续投影中的 UTF-16 结束下标，不包含该位置；差值不是 Unicode 码点字符数。
     */
    private final int endExclusive;

    private ProjectionRange(int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive <= startInclusive) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        this.startInclusive = startInclusive;
        this.endExclusive = endExclusive;
    }

    /**
     * 创建投影正文中的非空半开范围。
     */
    public static ProjectionRange of(int startInclusive, int endExclusive) {
        return new ProjectionRange(startInclusive, endExclusive);
    }

    /**
     * 判断同一投影坐标系内的两个范围是否相交。
     */
    public boolean overlaps(ProjectionRange other) {
        return other != null && startInclusive < other.endExclusive
                && other.startInclusive < endExclusive;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof ProjectionRange range)) {
            return false;
        }
        return startInclusive == range.startInclusive && endExclusive == range.endExclusive;
    }

    @Override
    public int hashCode() {
        return Objects.hash(startInclusive, endExclusive);
    }
}
