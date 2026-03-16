package com.fons.cloud.ai.rag.infrastructure.document.splitter;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TextSplitter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 重叠段落文本切分器
 * <pre>
 *     Spring AI 的切分方式不具备重叠（overlap）机制，也就是说相邻文本块之间没有共享内容。相比之下，LangChain / LangChain4j 通常提供 overlap 参数，
 *     用于在前一个文本块和下一个文本块之间保留一定数量的字符或 token，这样可以有效避免语义被切断，提高检索和召回的准确性。
 * </pre>
 * @author hongqy
 * @date 2026/3/12
 */
public class OverlapParagraphTextSplitter extends TextSplitter {
    // 每块最大字符数
    protected final int chunkSize;

    // 相邻块之间重叠字符数
    protected final int overlap;

    public OverlapParagraphTextSplitter(int chunkSize, int overlap) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize 必须大于 0");
        }
        if (overlap < 0) {
            throw new IllegalArgumentException("overlap 不能为负数");
        }
        if (overlap >= chunkSize) {
            throw new IllegalArgumentException("overlap 不能大于等于 chunkSize");
        }
        this.chunkSize = chunkSize;
        this.overlap = overlap;
    }

    @Override
    public List<Document> apply(List<Document> documents) {
        if (CollectionUtils.isEmpty(documents)) {
            return Collections.emptyList();
        }

        List<Document> result = new ArrayList<>();
        for (Document doc : documents) {
            List<String> chunks = splitText(doc.getText());
            for (String chunk : chunks) {
                result.add(new Document(chunk));
            }
        }
        return result;
    }

    @Override
    protected List<String> splitText(String text) {
        if (StringUtils.isBlank(text)) {
            return Collections.emptyList();
        }

        String[] paragraphs = text.split("\\n+");
        List<String> allChunks = new ArrayList<>();
        StringBuilder currentChunk = new StringBuilder();

        for (String paragraph : paragraphs) {
            if (StringUtils.isBlank(paragraph)) {
                continue;
            }

            int start = 0;
            while (start < paragraph.length()) {
                int remainingSpace = chunkSize - currentChunk.length();
                int end = Math.min(start + remainingSpace, paragraph.length());

                if (!currentChunk.isEmpty()) {
                    currentChunk.append("\n");
                }
                currentChunk.append(paragraph, start, end);

                // 如果当前块已满，保存并生成新块
                if (currentChunk.length() >= chunkSize) {
                    allChunks.add(currentChunk.toString());

                    // 计算重叠
                    String overlapText = "";
                    if (overlap > 0) {
                        int overlapStart = Math.max(0, currentChunk.length() - overlap);
                        overlapText = currentChunk.substring(overlapStart);
                    }

                    currentChunk = new StringBuilder();
                    if (!overlapText.isEmpty()) {
                        currentChunk.append(overlapText);
                    }
                }

                start = end;
            }
        }

        if (!currentChunk.isEmpty()) {
            allChunks.add(currentChunk.toString());
        }

        return allChunks;
    }

}
