package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

import java.util.Objects;

/**
 * 内容在原始文件中的来源位置。
 *
 * <p>构造器不公开，调用方必须使用具有明确业务含义的工厂方法。页面区域统一采用
 * {@code [0, 1]} 归一化坐标。</p>
 *
 * @author hongqy
 */
@Getter
public final class SourceLocation {

    /**
     * 来源位置类型。
     */
    public enum Type {
        /**
         * 仅确认所在页面，不声明页内区域。
         */
        PAGE,
        /**
         * 原始文本的 UTF-16 半开范围，不是检索投影范围。
         */
        TEXT_RANGE,
        /**
         * 确认所在页面及页内归一化矩形区域。
         */
        REGION,
        /**
         * 解析器确认的文件内部结构路径。
         */
        STRUCTURE,
        /**
         * 媒体中的毫秒半开范围。
         */
        TIME_RANGE,
        /**
         * 无法证明具体位置，不以估算坐标代替来源事实。
         */
        UNKNOWN
    }

    /**
     * 本位置实际具有的定位精度；其他类型的坐标字段为 null。
     */
    private final Type type;
    /**
     * 页码，从 1 开始；仅 PAGE、REGION 类型有值。
     */
    private final Integer pageNumber;
    /**
     * 原始文本起点，UTF-16 索引，从 0 开始且包含起点；仅 TEXT_RANGE 有值。
     */
    private final Integer startOffset;
    /**
     * 原始文本终点，UTF-16 索引且不包含终点；仅 TEXT_RANGE 有值。
     */
    private final Integer endOffset;
    /**
     * 页内矩形横向坐标，按页面宽度归一化到 [0, 1]；仅 REGION 有值。
     */
    private final Double x;
    /**
     * 页内矩形纵向坐标，按页面高度归一化到 [0, 1]；仅 REGION 有值。
     */
    private final Double y;
    /**
     * 矩形宽度，按页面宽度归一化，须为正数且不能越出页面。
     */
    private final Double width;
    /**
     * 矩形高度，按页面高度归一化，须为正数且不能越出页面。
     */
    private final Double height;
    /**
     * 解析器提供的实际结构定位路径；仅 STRUCTURE 有值，不替代文档分组关系。
     */
    private final String structuralPath;
    /**
     * 媒体起点，单位毫秒，包含起点；仅 TIME_RANGE 有值。
     */
    private final Long startMillis;
    /**
     * 媒体终点，单位毫秒，不包含终点；仅 TIME_RANGE 有值。
     */
    private final Long endMillis;

    private SourceLocation(
            Type type,
            Integer pageNumber,
            Integer startOffset,
            Integer endOffset,
            Double x,
            Double y,
            Double width,
            Double height,
            String structuralPath,
            Long startMillis,
            Long endMillis) {
        this.type = type;
        this.pageNumber = pageNumber;
        this.startOffset = startOffset;
        this.endOffset = endOffset;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.structuralPath = structuralPath;
        this.startMillis = startMillis;
        this.endMillis = endMillis;
    }

    /**
     * 创建页级位置，页码从一开始。
     */
    public static SourceLocation page(int pageNumber) {
        require(pageNumber > 0);
        return new SourceLocation(Type.PAGE, pageNumber, null, null,
                null, null, null, null, null, null, null);
    }

    /**
     * 创建原始文本 UTF-16 半开范围。
     */
    public static SourceLocation textRange(int startInclusive, int endExclusive) {
        require(startInclusive >= 0 && endExclusive > startInclusive);
        return new SourceLocation(Type.TEXT_RANGE, null, startInclusive, endExclusive,
                null, null, null, null, null, null, null);
    }

    /**
     * 创建页内归一化矩形区域。
     */
    public static SourceLocation region(
            int pageNumber, double x, double y, double width, double height) {
        boolean finite = Double.isFinite(x) && Double.isFinite(y)
                && Double.isFinite(width) && Double.isFinite(height);
        require(pageNumber > 0 && finite && x >= 0 && y >= 0 && width > 0 && height > 0
                && x + width <= 1 && y + height <= 1);
        return new SourceLocation(Type.REGION, pageNumber, null, null,
                x, y, width, height, null, null, null);
    }

    /**
     * 创建文件内部结构路径，例如 sheet/cell、slide/shape 或 paragraph。
     */
    public static SourceLocation structure(String structuralPath) {
        require(structuralPath != null && !structuralPath.isBlank());
        return new SourceLocation(Type.STRUCTURE, null, null, null,
                null, null, null, null, structuralPath, null, null);
    }

    /**
     * 创建音视频毫秒半开范围。
     */
    public static SourceLocation timeRange(long startInclusive, long endExclusive) {
        require(startInclusive >= 0 && endExclusive > startInclusive);
        return new SourceLocation(Type.TIME_RANGE, null, null, null,
                null, null, null, null, null, startInclusive, endExclusive);
    }

    /**
     * 创建无法证明具体位置的来源。
     */
    public static SourceLocation unknown() {
        return new SourceLocation(Type.UNKNOWN, null, null, null,
                null, null, null, null, null, null, null);
    }

    /**
     * 判断来源位置是否已经确认。
     */
    public boolean isKnown() {
        return type != Type.UNKNOWN;
    }

    /**
     * 判断是否属于页面坐标。
     */
    public boolean isPageBased() {
        return type == Type.PAGE || type == Type.REGION;
    }

    /**
     * 判断是否位于指定页面。
     */
    public boolean belongsToPage(int candidatePageNumber) {
        return pageNumber != null && pageNumber == candidatePageNumber;
    }

    /**
     * 判断两个同坐标系位置是否重叠。
     */
    public boolean overlaps(SourceLocation other) {
        if (other == null || type != other.type) {
            return false;
        }
        return switch (type) {
            case PAGE -> pageNumber.equals(other.pageNumber);
            case TEXT_RANGE -> startOffset < other.endOffset && other.startOffset < endOffset;
            case REGION -> pageNumber.equals(other.pageNumber)
                    && x < other.x + other.width && other.x < x + width
                    && y < other.y + other.height && other.y < y + height;
            case STRUCTURE -> structuralPath.equals(other.structuralPath);
            case TIME_RANGE -> startMillis < other.endMillis && other.startMillis < endMillis;
            case UNKNOWN -> false;
        };
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof SourceLocation location)) {
            return false;
        }
        return type == location.type
                && Objects.equals(pageNumber, location.pageNumber)
                && Objects.equals(startOffset, location.startOffset)
                && Objects.equals(endOffset, location.endOffset)
                && Objects.equals(x, location.x)
                && Objects.equals(y, location.y)
                && Objects.equals(width, location.width)
                && Objects.equals(height, location.height)
                && Objects.equals(structuralPath, location.structuralPath)
                && Objects.equals(startMillis, location.startMillis)
                && Objects.equals(endMillis, location.endMillis);
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, pageNumber, startOffset, endOffset,
                x, y, width, height, structuralPath, startMillis, endMillis);
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
    }
}
