package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.constants.AgentType;
import org.junit.jupiter.api.Test;
import org.redisson.api.RBucket;
import org.redisson.api.RTopic;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
import reactor.core.publisher.Sinks;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentTaskManagerCompletionTest {

    @SuppressWarnings("unchecked")
    @Test
    void shouldReleaseNormallyWithoutEmittingStopMessage() {
        RedissonClient redissonClient = mock(RedissonClient.class);
        RTopic topic = mock(RTopic.class);
        RBucket<String> bucket = mock(RBucket.class);
        Sinks.Many<String> sink = mock(Sinks.Many.class);
        when(redissonClient.getTopic(anyString())).thenReturn(topic);
        doReturn(bucket).when(redissonClient).getBucket(anyString(), any(StringCodec.class));
        when(bucket.setIfAbsent(anyString(), any(Duration.class))).thenReturn(true);
        when(bucket.compareAndSet(anyString(), org.mockito.ArgumentMatchers.isNull())).thenReturn(true);
        AgentTaskManager manager = new AgentTaskManager("instance-a", redissonClient);
        AgentTaskHandle handle = new AgentTaskHandle("conversation", "run-1");

        assertTrue(manager.registerTask(handle, sink, AgentType.SKILLS).isSuccess());
        assertTrue(manager.completeTask(handle));
        assertFalse(manager.completeTask(handle));

        verify(bucket).compareAndSet(anyString(), org.mockito.ArgumentMatchers.isNull());
        verify(sink, never()).tryEmitNext(anyString());
        verify(sink, never()).tryEmitComplete();
    }
}
