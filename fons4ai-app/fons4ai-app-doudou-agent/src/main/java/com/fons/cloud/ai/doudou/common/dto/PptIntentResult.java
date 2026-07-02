package com.fons.cloud.ai.doudou.common.dto;

import com.fons.cloud.ai.doudou.common.constants.PptIntent;

/**
 * PPT意图结果
 * @param intent 意图类型
 * @param reason 原因
 * @author hongqy
 */
public record PptIntentResult(PptIntent intent, String reason) {

}
