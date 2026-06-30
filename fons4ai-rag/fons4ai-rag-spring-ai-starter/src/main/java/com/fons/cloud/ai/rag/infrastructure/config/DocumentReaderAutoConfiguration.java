package com.fons.cloud.ai.rag.infrastructure.config;

import com.fons.cloud.ai.rag.document.reader.DocumentReaderFacade;
import com.fons.cloud.ai.rag.document.reader.DocumentReaderStrategy;
import com.fons.cloud.ai.rag.document.reader.support.*;
import com.fons.cloud.ai.rag.infrastructure.multiplemodal.MultipleModalChatModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

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
    public DocumentReaderStrategy imageReaderStrategy(MultipleModalChatModel multipleModalChatModel) {
        return new ImageReadStrategy(multipleModalChatModel);
    }

    @Bean
    DocumentReaderFacade documentReaderFacade() {
        return new DocumentReaderFacade();
    }



}
