package com.fons.cloud.ai.rag.model.parsing;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * 文档解析器选择策略。
 *
 * <p>显式选择和自动选择通过有含义的工厂方法创建，不暴露 mode 与 providerId 的非法组合。</p>
 *
 * @author hongqy
 */
@Getter
public final class ParserSelectionPolicy {

    /** 显式指定的解析器标识；null 表示自动选择，显式选择失败时不得换用其他解析器。 */
    private final String requiredParserId;
    /** 自动选择的候选范围，空集表示不限；仅表达选择约束，不证明解析器可用或支持格式。 */
    private final Set<String> allowedParserIds;

    private ParserSelectionPolicy(String requiredParserId, Set<String> allowedParserIds) {
        this.requiredParserId = normalize(requiredParserId);
        this.allowedParserIds = copyAndValidate(allowedParserIds);
        if (this.requiredParserId != null && !this.allowedParserIds.isEmpty()
                && !this.allowedParserIds.contains(this.requiredParserId)) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
    }

    /** 创建不限制 Parser 范围的自动选择策略。 */
    public static ParserSelectionPolicy automatic() {
        return new ParserSelectionPolicy(null, Set.of());
    }

    /** 创建限制候选 Parser 范围的自动选择策略。 */
    public static ParserSelectionPolicy automatic(Set<String> allowedParserIds) {
        return new ParserSelectionPolicy(null, allowedParserIds);
    }

    /** 创建禁止降级的显式 Parser 选择策略。 */
    public static ParserSelectionPolicy use(String parserId) {
        String requiredId = normalize(parserId);
        if (!validParserId(requiredId)) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        return new ParserSelectionPolicy(requiredId, Set.of(requiredId));
    }

    /** 判断当前策略是否显式指定了解析器。 */
    public boolean isExplicit() {
        return requiredParserId != null;
    }

    /** 判断一个解析器是否允许参加选择。 */
    public boolean allows(String parserId) {
        String candidate = normalize(parserId);
        if (candidate == null) {
            return false;
        }
        if (requiredParserId != null) {
            return requiredParserId.equals(candidate);
        }
        return allowedParserIds.isEmpty() || allowedParserIds.contains(candidate);
    }

    /** 校验最终选择结果是否符合策略。 */
    public void assertAllows(String parserId) {
        if (!allows(parserId)) {
            throw BusinessRuntimeException.of(RagResultCode.DOCUMENT_PARSER_NOT_FOUND);
        }
    }

    private static Set<String> copyAndValidate(Set<String> parserIds) {
        if (parserIds == null || parserIds.isEmpty()) {
            return Set.of();
        }
        LinkedHashSet<String> copy = new LinkedHashSet<>();
        for (String parserId : parserIds) {
            String normalized = normalize(parserId);
            if (!validParserId(normalized)) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
            copy.add(normalized);
        }
        return Set.copyOf(copy);
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    private static boolean validParserId(String parserId) {
        return parserId != null && parserId.matches("[a-z0-9]+(?:[.-][a-z0-9]+)+");
    }
}
