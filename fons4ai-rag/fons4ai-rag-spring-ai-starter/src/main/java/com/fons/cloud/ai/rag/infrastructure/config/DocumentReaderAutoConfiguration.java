package com.fons.cloud.ai.rag.infrastructure.config;

import com.fons.cloud.ai.capability.multimodal.ImageRecognitionService;
import com.fons.cloud.ai.rag.document.reader.DocumentReaderFacade;
import com.fons.cloud.ai.rag.document.reader.DocumentReaderStrategy;
import com.fons.cloud.ai.rag.document.reader.support.*;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * @author hongqy
 */
@Configuration
public class DocumentReaderAutoConfiguration {

    @Bean
    public DocumentReaderStrategy jsonReaderStrategy() {
        return new JsonReaderStrategy();
    }

    @Bean
    public DocumentReaderStrategy markdownReaderStrategy() {
        return new MarkdownReaderStrategy();
    }

    @Bean
    public DocumentReaderStrategy pdfReaderStrategy() {
        return new PdfReaderStrategy();
    }

    @Bean
    public DocumentReaderStrategy textReaderStrategy() {
        return new TextReaderStrategy();
    }

    @Bean
    public DocumentReaderStrategy documentReaderStrategy() {
        return new DocReaderStrategy();
    }

    @Bean
    @ConditionalOnBean(ImageRecognitionService.class)
    public DocumentReaderStrategy imageReaderStrategy(ImageRecognitionService imageRecognitionService) {
        return new ImageReadStrategy(imageRecognitionService);
    }

    @Bean
    DocumentReaderFacade documentReaderFacade(List<DocumentReaderStrategy> strategies) {
        return new DocumentReaderFacade(strategies);
    }



}
