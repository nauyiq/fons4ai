package com.fons.cloud.ai.doudou.controller;

import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 智能体控制器
 * @author hongqy
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/doudou/agent")
@Tag(name = "智能体管理", description = "提供网页搜索、文件问答和PPT生成的流式接口")
public class AgentController {




}
