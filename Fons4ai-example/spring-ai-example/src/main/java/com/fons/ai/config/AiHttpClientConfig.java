package com.fons.ai.config;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ClientHttpConnector;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

/**
 * @author hongqy
 * @date 2026/3/2
 */
@Configuration
public class AiHttpClientConfig {

    @Value("${spring.ai.client.timeout.connect:50000}")
    private int connectTimeoutMs;

    @Value("${spring.ai.client.timeout.read:30000}")
    private int readTimeoutMs;

    @Bean
    @ConditionalOnMissingBean
    public WebClient.Builder webClientBuilder() {
        HttpClient httpClient = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, connectTimeoutMs)
                .responseTimeout(Duration.ofMillis(readTimeoutMs))
                .doOnConnected(conn -> conn
                        .addHandlerLast(new ReadTimeoutHandler(readTimeoutMs, TimeUnit.MILLISECONDS)));

        ClientHttpConnector connector = new ReactorClientHttpConnector(httpClient);
        return WebClient.builder().clientConnector(connector);
    }






}
