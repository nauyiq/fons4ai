package com.fons.cloud.ai.doudou.common.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * PPT意图
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum PptIntent {

    CREATE_PPT("CREATE_PPT", "新建PPT"),

    MODIFY_PPT("MODIFY_PPT", "修改PPT"),

    RESUME_PPT("RESUME_PPT", "断点重连（继续之前失败的任务）")

    ;
    private final String code;
    private final String desc;


}
