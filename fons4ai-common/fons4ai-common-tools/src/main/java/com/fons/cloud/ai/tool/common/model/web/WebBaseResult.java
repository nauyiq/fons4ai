package com.fons.cloud.ai.tool.common.model.web;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

import java.io.Serial;
import java.io.Serializable;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@SuperBuilder
public class WebBaseResult implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 地址
     */
    private String url;

}
