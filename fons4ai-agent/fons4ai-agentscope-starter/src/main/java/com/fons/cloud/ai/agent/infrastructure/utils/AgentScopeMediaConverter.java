package com.fons.cloud.ai.agent.infrastructure.utils;

import com.fons.cloud.ai.agent.api.AgentScopeMediaUriResolver;
import com.fons.cloud.ai.agent.core.AgentScopeRunContext;
import com.fons.cloud.ai.agent.model.response.AgentMediaInfo;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.DataBlock;
import io.agentscope.core.message.Source;
import io.agentscope.core.message.URLSource;
import org.apache.commons.lang3.StringUtils;

import java.net.URI;
import java.net.URLConnection;

/**
 * AgentScope完整媒体结果转换器。
 *
 * <p>只转换最终消息中的DataBlock，不处理工具中间结果或媒体流式分片。</p>
 *
 * @author hongqy
 */
public final class AgentScopeMediaConverter {

    private static final AgentScopeMediaConverter INSTANCE = new AgentScopeMediaConverter();

    private AgentScopeMediaConverter() {
    }

    /**
     * 获取媒体转换器单例。
     *
     * @return 媒体转换器
     */
    public static AgentScopeMediaConverter getInstance() {
        return INSTANCE;
    }

    /**
     * 将一项AgentScope最终媒体转换为common媒体信息。
     *
     * @param context 当前Run上下文
     * @param block AgentScope媒体内容块
     * @param resolver 非公开资源的URI解析器，可为null
     * @return common完整媒体信息
     */
    public AgentMediaInfo convert(AgentScopeRunContext context,
                                  DataBlock block,
                                  AgentScopeMediaUriResolver resolver) {
        Source source = block.getSource();
        String uri = resolveUri(context, block, source, resolver);
        String mimeType = resolveMimeType(block, source);
        return AgentMediaInfo.builder()
                .mediaId(block.getId())
                .mimeType(mimeType)
                .uri(uri)
                .name(block.getName())
                .build();
    }

    /**
     * HTTP(S)资源直接透传，其余资源由下游保存并提供可访问地址。
     */
    private String resolveUri(AgentScopeRunContext context,
                              DataBlock block,
                              Source source,
                              AgentScopeMediaUriResolver resolver) {
        if (source instanceof URLSource urlSource && isHttpUrl(urlSource.getUrl())) {
            return urlSource.getUrl();
        }
        if (resolver == null) {
            throw SystemIntervalException.of("AgentScope media requires a URI resolver");
        }
        String uri = resolver.resolve(context, block);
        if (StringUtils.isBlank(uri)) {
            throw SystemIntervalException.of("AgentScope media URI resolver returned an empty URI");
        }
        return uri;
    }

    private boolean isHttpUrl(String uri) {
        try {
            URI parsed = URI.create(uri);
            String scheme = parsed.getScheme();
            return ("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme))
                    && StringUtils.isNotBlank(parsed.getHost());
        } catch (IllegalArgumentException exception) {
            return false;
        }
    }

    /**
     * 优先使用原生MIME提示；缺失时通过资源名称推断。
     */
    private String resolveMimeType(DataBlock block, Source source) {
        if (source instanceof Base64Source base64Source) {
            return base64Source.getMediaType();
        }
        if (source instanceof URLSource urlSource && StringUtils.isNotBlank(urlSource.getMimeType())) {
            return urlSource.getMimeType();
        }
        String name = block.getName();
        if (StringUtils.isBlank(name) && source instanceof URLSource urlSource) {
            try {
                name = URI.create(urlSource.getUrl()).getPath();
            } catch (IllegalArgumentException exception) {
                name = null;
            }
        }
        String inferred = name == null ? null : URLConnection.guessContentTypeFromName(name);
        return StringUtils.defaultIfBlank(inferred, "application/octet-stream");
    }
}
