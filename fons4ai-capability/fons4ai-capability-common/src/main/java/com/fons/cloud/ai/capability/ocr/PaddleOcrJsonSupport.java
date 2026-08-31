package com.fons.cloud.ai.capability.ocr;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONException;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * PaddleOCR HTTP 协议的 JSON 读写与字段校验支持。
 * <p>
 * 使用 fastjson2，且仅保留与远端协议相关的结构校验。
 */
public final class PaddleOcrJsonSupport {

    private PaddleOcrJsonSupport() {
    }

    /**
     * 将请求对象编码为 JSON。
     *
     * @param source 待编码对象
     * @return JSON 文本
     */
    public static String toJson(Map<String, Object> source) {
        try {
            return JSON.toJSONString(source);
        } catch (JSONException exception) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED.getCode(), exception);
        }
    }

    /**
     * 解析顶层 JSON 对象。
     *
     * @param body JSON 文本
     * @return JSON 对象
     */
    public static Map<String, Object> parseObject(String body) {
        if (body == null || body.isBlank()) {
            throw new IllegalArgumentException("JSON 不可为空");
        }
        try {
            JSONObject object = JSON.parseObject(body);
            if (object == null) {
                throw new IllegalArgumentException("JSON 顶层必须是对象");
            }
            return object;
        } catch (JSONException exception) {
            throw new IllegalArgumentException("JSON 顶层必须是对象", exception);
        }
    }

    /**
     * 将 JSON 值转换为对象。
     */
    public static Map<String, Object> asObject(Object value, String field) {
        if (!(value instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException("JSON 字段 " + field + " 必须是对象");
        }
        Map<String, Object> result = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!(entry.getKey() instanceof String key)) {
                throw new IllegalArgumentException("JSON 对象键必须为字符串");
            }
            result.put(key, entry.getValue());
        }
        return result;
    }

    /**
     * 读取对象字段并要求其为嵌套对象。
     */
    public static Map<String, Object> requiredObject(Map<String, Object> source, String field) {
        return asObject(source.get(field), field);
    }

    /**
     * 读取对象字段并要求其为数组。
     */
    public static List<Object> requiredArray(Map<String, Object> source, String field) {
        Object value = source.get(field);
        if (value instanceof List<?> list) {
            return List.copyOf(list);
        }
        throw new IllegalArgumentException("JSON 字段 " + field + " 必须是数组");
    }

    /**
     * 读取非空字符串字段。
     */
    public static String requiredString(Map<String, Object> source, String field) {
        Object value = source.get(field);
        if (value instanceof String text && !text.isBlank()) {
            return text;
        }
        throw new IllegalArgumentException("JSON 字段 " + field + " 必须是非空字符串");
    }

    /**
     * 将 JSON 数字转换为 int。
     */
    public static int requiredInt(Map<String, Object> source, String field) {
        Object value = source.get(field);
        if (value instanceof Number number) {
            return number.intValue();
        }
        throw new IllegalArgumentException("JSON 字段 " + field + " 必须是数字");
    }
}
