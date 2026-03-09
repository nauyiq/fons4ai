package com.fons.cloud.ai.function;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Description;

import java.util.function.Function;

/**
 * @author hongqy
 * @date 2026/3/5
 */
@Configuration
public class FunctionCallConfiguration {

    @Bean
    @Description("根据用户输入的时区获取该时区的当前时间")
    public Function<com.fons.cloud.ai.function.TimeService.Request, com.fons.cloud.ai.function.TimeService.Response> getTimeFunction(com.fons.cloud.ai.function.TimeService timeService) {
        return timeService::getTimeByZoneId;
    }

}
