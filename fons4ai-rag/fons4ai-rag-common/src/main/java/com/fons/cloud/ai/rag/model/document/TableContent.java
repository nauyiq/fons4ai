package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * 不退化为 Markdown 字符串的结构化表格。
 *
 * @author hongqy
 */
@Getter
public final class TableContent {

    /**
     * 表格实际总行数，至少为 1；不是分块时的每块行数限制。
     */
    private final int rowCount;
    /**
     * 表格实际总列数，至少为 1。
     */
    private final int columnCount;
    /**
     * 按行列排序的单元格，只保存合并单元格的锚点；允许稀疏，不推断首行为表头。
     */
    private final List<Cell> cells;
    /** 解析器确认的表头事实；默认未知，不能把第一行自动当作表头。 */
    private final Header header;

    private TableContent(Builder builder) {
        this.rowCount = builder.rowCount;
        this.columnCount = builder.columnCount;
        this.cells = List.copyOf(builder.cells);
        this.header = builder.header;
    }

    /**
     * 创建指定尺寸的表格构建器。
     */
    public static Builder builder(int rowCount, int columnCount) {
        return new Builder(rowCount, columnCount);
    }

    /**
     * 查找指定坐标所属的单元格。
     */
    public Optional<Cell> cellAt(int rowIndex, int columnIndex) {
        return cells.stream()
                .filter(cell -> cell.contains(rowIndex, columnIndex))
                .findFirst();
    }

    /**
     * 按行列顺序生成用于检索的纯文本。
     */
    public String plainText() {
        return renderRows(0, rowCount);
    }

    /** 返回真实表头单元格；未知和确认无表头都返回空列表，但状态由 Header 明确区分。 */
    public List<Cell> headerCells() {
        return header.getState() == Header.State.IDENTIFIED
                ? cellsInRows(header.getStartRowInclusive(), header.getEndRowExclusive()) : List.of();
    }

    /** 只呈现已经确认的表头，供分块作为单独的 CONTEXT 片段携带。 */
    public String headerText() {
        return header.getState() == Header.State.IDENTIFIED
                ? renderRows(header.getStartRowInclusive(), header.getEndRowExclusive()) : "";
    }

    /** 按跨行合并关系计算不可任意拆开的数据行组，已确认的表头不作为新数据行。 */
    public List<RowGroup> logicalRowGroups() {
        List<RowGroup> groups = new ArrayList<>();
        if (header.getState() == Header.State.IDENTIFIED) {
            addLogicalGroups(groups, 0, header.getStartRowInclusive());
            addLogicalGroups(groups, header.getEndRowExclusive(), rowCount);
        } else {
            addLogicalGroups(groups, 0, rowCount);
        }
        return List.copyOf(groups);
    }

    /**
     * 在同一张表内按完整逻辑行组打包，输出携带已确认表头，字符上限包含表头和分隔符。
     *
     * <p>只提供中立表格行为，不读取文件、不创建 Chunk，也不注册技术分块策略。
     * 不能安全容纳一个逻辑行组时明确失败，长字段细分及精确分块来源由后续阶段处理。</p>
     */
    public R<List<RowGroup>> splitRows(int maximumCharacters, boolean requireHeader) {
        if (maximumCharacters < 1) {
            return R.failed(RagResultCode.CHUNKING_POLICY_INVALID);
        }
        if (requireHeader && header.getState() != Header.State.IDENTIFIED) {
            return R.failed(RagResultCode.CHUNKING_DOCUMENT_UNSUPPORTED);
        }
        List<RowGroup> units = logicalRowGroups();
        if (units.isEmpty()) {
            return R.failed(RagResultCode.CHUNKING_CONTENT_EMPTY);
        }
        long contextSize = header.getState() == Header.State.IDENTIFIED
                ? characterCount(header.getStartRowInclusive(), header.getEndRowExclusive()) + 1 : 0;
        List<RowGroup> output = new ArrayList<>();
        RowGroup pending = null;
        long pendingSize = 0;
        for (RowGroup unit : units) {
            long unitSize = unit.getBodyCharacterCount();
            // 先测量再呈现，避免为了发现超限而先展开一个巨大表格字符串。
            if (contextSize + unitSize > maximumCharacters) {
                return R.failed(RagResultCode.CHUNKING_UNIT_TOO_LARGE);
            }
            if (pending != null && pending.getEndRowExclusive() == unit.getStartRowInclusive()
                    && pendingSize + 1 + unitSize <= maximumCharacters) {
                pending = new RowGroup(this, pending.getStartRowInclusive(), unit.getEndRowExclusive(),
                        pending.getBodyCharacterCount() + 1 + unitSize);
                pendingSize += 1 + unitSize;
            } else {
                if (pending != null) {
                    output.add(pending);
                }
                pending = unit;
                pendingSize = contextSize + unitSize;
            }
        }
        output.add(pending);
        return R.success(List.copyOf(output));
    }

    private void addLogicalGroups(List<RowGroup> groups, int start, int end) {
        int cellIndex = 0;
        while (cellIndex < cells.size() && cells.get(cellIndex).getRowIndex() < start) {
            cellIndex++;
        }
        int row = start;
        while (row < end) {
            int groupEnd = row + 1;
            long cellCharacters = 0;
            // 跨行范围传递合并：后续行上的另一个跨行单元格可以继续延长同一逻辑组。
            while (cellIndex < cells.size() && cells.get(cellIndex).getRowIndex() < groupEnd) {
                Cell cell = cells.get(cellIndex++);
                groupEnd = Math.max(groupEnd, cell.getRowIndex() + cell.getRowSpan());
                cellCharacters += cell.getText().codePointCount(0, cell.getText().length());
            }
            long bodySize = (long) (groupEnd - row) * (columnCount - 1)
                    + (groupEnd - row - 1) + cellCharacters;
            groups.add(new RowGroup(this, row, groupEnd, bodySize));
            row = groupEnd;
        }
    }

    private List<Cell> cellsInRows(int start, int end) {
        return cells.stream().filter(cell -> cell.getRowIndex() >= start && cell.getRowIndex() < end).toList();
    }

    private long characterCount(int start, int end) {
        long count = (long) (end - start) * (columnCount - 1) + Math.max(0, end - start - 1);
        for (Cell cell : cells) {
            if (cell.getRowIndex() >= start && cell.getRowIndex() < end) {
                count += cell.getText().codePointCount(0, cell.getText().length());
            }
        }
        return count;
    }

    private String renderRows(int start, int end) {
        StringBuilder text = new StringBuilder();
        int cellIndex = 0;
        while (cellIndex < cells.size() && cells.get(cellIndex).getRowIndex() < start) {
            cellIndex++;
        }
        // 按已排序的锚点一次向前呈现，不再对每个坐标扫描全部单元格。
        for (int row = start; row < end; row++) {
            int column = 0;
            while (cellIndex < cells.size() && cells.get(cellIndex).getRowIndex() == row) {
                Cell cell = cells.get(cellIndex++);
                text.append("\t".repeat(cell.getColumnIndex() - column));
                text.append(cell.getText());
                column = cell.getColumnIndex();
            }
            text.append("\t".repeat(columnCount - 1 - column));
            if (row + 1 < end) {
                text.append('\n');
            }
        }
        return text.toString();
    }

    /**
     * 表格构建器，负责阻止越界或相互覆盖的单元格进入表格。
     *
     * @author hongqy
     */
    public static final class Builder {

        /**
         * 待构建表格的行边界，新增单元格的跨行范围不得越界。
         */
        private final int rowCount;
        /**
         * 待构建表格的列边界，新增单元格的跨列范围不得越界。
         */
        private final int columnCount;
        /**
         * 已确认且互不覆盖的单元格，构建时排序并复制为不可变列表。
         */
        private final List<Cell> cells = new ArrayList<>();
        /** 未声明时保持未知；显式声明只保存事实，build 时复核范围和合并关系。 */
        private Header header = new Header(Header.State.UNKNOWN, 0, 0);

        private Builder(int rowCount, int columnCount) {
            if (rowCount < 1 || columnCount < 1) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
            this.rowCount = rowCount;
            this.columnCount = columnCount;
        }

        /**
         * 添加普通单元格。
         */
        public Builder addCell(int rowIndex, int columnIndex, String text) {
            return addCell(rowIndex, columnIndex, 1, 1, text);
        }

        /**
         * 添加可能跨行或跨列的单元格。
         */
        public Builder addCell(
                int rowIndex, int columnIndex, int rowSpan, int columnSpan, String text) {
            return addCell(rowIndex, columnIndex, rowSpan, columnSpan, text, SourceLocation.unknown());
        }

        /** 添加已确认原文件位置的普通单元格；行列索引本身不当作原文件定位。 */
        public Builder addCell(int rowIndex, int columnIndex, String text, SourceLocation sourceLocation) {
            return addCell(rowIndex, columnIndex, 1, 1, text, sourceLocation);
        }

        /** 添加合并单元格及适配器实际证明的位置，未知位置不补猜。 */
        public Builder addCell(int rowIndex, int columnIndex, int rowSpan, int columnSpan,
                               String text, SourceLocation sourceLocation) {
            Cell candidate = new Cell(rowIndex, columnIndex, rowSpan, columnSpan, text, sourceLocation);
            // 用 long 计算覆盖终点，避免 int 溢出后让越界单元格通过验收。
            if ((long) candidate.getRowIndex() + candidate.getRowSpan() > rowCount
                    || (long) candidate.getColumnIndex() + candidate.getColumnSpan() > columnCount
                    || cells.stream().anyMatch(existing -> existing.overlaps(candidate))) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
            cells.add(candidate);
            return this;
        }

        /** 尚未可靠识别表头，不等于已经确认没有表头。 */
        public Builder headerUnknown() {
            header = new Header(Header.State.UNKNOWN, 0, 0);
            return this;
        }

        /** 明确确认该表没有表头。 */
        public Builder withoutHeader() {
            header = new Header(Header.State.ABSENT, 0, 0);
            return this;
        }

        /** 声明真实的连续表头范围，表内索引从零开始，结束行不包含在范围中。 */
        public Builder headerRows(int startRowInclusive, int endRowExclusive) {
            if (startRowInclusive < 0 || endRowExclusive <= startRowInclusive || endRowExclusive > rowCount) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
            header = new Header(Header.State.IDENTIFIED, startRowInclusive, endRowExclusive);
            return this;
        }

        /**
         * 按稳定行列顺序构建不可变表格。
         */
        public TableContent build() {
            if (header.getState() == Header.State.IDENTIFIED) {
                for (Cell cell : cells) {
                    int end = cell.getRowIndex() + cell.getRowSpan();
                    // 表头两侧均是保护边界，不能把一个跨行单元格截成表头和数据两半。
                    if (crosses(cell.getRowIndex(), end, header.getStartRowInclusive())
                            || crosses(cell.getRowIndex(), end, header.getEndRowExclusive())) {
                        throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
                    }
                }
            }
            cells.sort(Comparator.comparingInt(Cell::getRowIndex)
                    .thenComparingInt(Cell::getColumnIndex));
            return new TableContent(this);
        }

        private static boolean crosses(int start, int end, int boundary) {
            return start < boundary && end > boundary;
        }
    }

    /** 表头事实值对象，不用一个空列表混同未知和确认不存在。 */
    @Getter
    public static final class Header {
        public enum State {
            /** 未可靠识别。 */
            UNKNOWN,
            /** 已确认没有表头。 */
            ABSENT,
            /** 已确认连续表头行范围。 */
            IDENTIFIED
        }

        /** 解析器声明且已校验的状态。 */
        private final State state;
        /** IDENTIFIED 时的表内起始行，包含该行；其他状态为零且无范围含义。 */
        private final int startRowInclusive;
        /** IDENTIFIED 时的表内结束行，不包含该行；其他状态为零且无范围含义。 */
        private final int endRowExclusive;

        private Header(State state, int startRowInclusive, int endRowExclusive) {
            this.state = state;
            this.startRowInclusive = startRowInclusive;
            this.endRowExclusive = endRowExclusive;
        }
    }

    /** 一个同表内、未切断合并单元格的数据行范围；表头作为上下文，不改变数据行索引。 */
    @Getter
    public static final class RowGroup {
        /** 绑定真实表格，派生的单元格均来自此不可变内容对象。 */
        private final TableContent table;
        /** 实际数据起始行，表内从零开始，不是 Excel 原文件行号。 */
        private final int startRowInclusive;
        /** 实际数据结束行，不包含该行；范围不包含需要重复携带的表头。 */
        private final int endRowExclusive;

        /** 从真实单元格及分隔符推导的正文码点字符数，不含重复表头，避免打包时反复扫描整表。 */
        private final long bodyCharacterCount;

        private RowGroup(TableContent table, int startRowInclusive, int endRowExclusive, long bodyCharacterCount) {
            this.table = table;
            this.startRowInclusive = startRowInclusive;
            this.endRowExclusive = endRowExclusive;
            this.bodyCharacterCount = bodyCharacterCount;
        }

        /** 返回正文部分的真实锚点单元格；跨行单元格不会被截断。 */
        public List<Cell> bodyCells() {
            return table.cellsInRows(startRowInclusive, endRowExclusive);
        }

        /** 返回重复携带的真实表头，后续来源组装可与数据行明确区分。 */
        public List<Cell> headerCells() {
            return table.headerCells();
        }

        /** 数据正文不混入重复表头，表头由 ChunkSet 另建上下文片段。 */
        public String bodyText() {
            return table.renderRows(startRowInclusive, endRowExclusive);
        }

        /** 呈现完整输出，表头、制表符和换行均计入字符限制。 */
        public String plainText() {
            String body = table.renderRows(startRowInclusive, endRowExclusive);
            return table.header.getState() == Header.State.IDENTIFIED
                    ? table.renderRows(table.header.getStartRowInclusive(), table.header.getEndRowExclusive())
                        + "\n" + body : body;
        }
    }

    /**
     * 表格单元格。
     *
     * @author hongqy
     */
    public static final class Cell {

        /**
         * 锚点行索引，从 0 开始；合并单元格使用其起始行。
         */
        private final int rowIndex;
        /**
         * 锚点列索引，从 0 开始；合并单元格使用其起始列。
         */
        private final int columnIndex;
        /**
         * 覆盖的行数，至少为 1；普通单元格为 1。
         */
        private final int rowSpan;
        /**
         * 覆盖的列数，至少为 1；普通单元格为 1。
         */
        private final int columnSpan;
        /**
         * 解析器确认的单元格文本；null 归一化为空串，不在此执行公式计算。
         */
        private final String text;
        /** 适配器实际提供的单元格原文件位置；未确认时为 UNKNOWN，不根据表内索引伪造。 */
        private final SourceLocation sourceLocation;

        private Cell(int rowIndex, int columnIndex, int rowSpan, int columnSpan,
                     String text, SourceLocation sourceLocation) {
            if (rowIndex < 0 || columnIndex < 0 || rowSpan < 1 || columnSpan < 1) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
            this.rowIndex = rowIndex;
            this.columnIndex = columnIndex;
            this.rowSpan = rowSpan;
            this.columnSpan = columnSpan;
            this.text = text == null ? "" : text;
            this.sourceLocation = sourceLocation == null ? SourceLocation.unknown() : sourceLocation;
        }

        private boolean contains(int row, int column) {
            return row >= rowIndex && row < rowIndex + rowSpan
                    && column >= columnIndex && column < columnIndex + columnSpan;
        }

        private boolean overlaps(Cell other) {
            return rowIndex < other.rowIndex + other.rowSpan
                    && other.rowIndex < rowIndex + rowSpan
                    && columnIndex < other.columnIndex + other.columnSpan
                    && other.columnIndex < columnIndex + columnSpan;
        }

        public int getRowIndex() {
            return rowIndex;
        }

        public int getColumnIndex() {
            return columnIndex;
        }

        public int getRowSpan() {
            return rowSpan;
        }

        public int getColumnSpan() {
            return columnSpan;
        }

        public String getText() {
            return text;
        }

        public SourceLocation getSourceLocation() {
            return sourceLocation;
        }
    }
}
