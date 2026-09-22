package com.fons.cloud.ai.rag.model.document;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * 一条对象记录的真实内容树。格式库负责读取文件，本类只保存、查询和呈现已经确认的事实。
 * <p>路径使用从根开始的子节点序号，空路径表示根；保留重复名称和 XML 混合内容的源顺序，
 * 不把展示路径冒充原文件字符位置，也不承诺还原原始文件字节。</p>
 */
@Getter
public final class RecordContent {
    /** 本条记录的唯一根节点；树内同一个节点对象不能出现在两个位置。 */
    private final Node root;

    private RecordContent(Node root) {
        require(root != null);
        Set<Node> members = Collections.newSetFromMap(new IdentityHashMap<>());
        validateTree(root, members);
        this.root = root;
    }

    public static RecordContent of(Node root) {
        return new RecordContent(root);
    }

    /** 按真实子节点序号查询，不按名称猜测重复字段属于哪一个节点。 */
    public Node nodeAt(List<Integer> path) {
        require(path != null);
        Node current = root;
        for (Integer index : path) {
            require(index != null && index >= 0 && index < current.children.size());
            current = current.children.get(index);
        }
        return current;
    }

    /** 查找本树中真实节点对象的路径，字段相同的外部对象不视为本树成员。 */
    public List<Integer> pathOf(Node node) {
        List<Integer> path = new ArrayList<>();
        require(node != null && findPath(root, node, path));
        return List.copyOf(path);
    }

    /** 返回源顺序中的直接子节点；具体算法决定哪些子节点可独立分块。 */
    public List<Node> childrenAt(List<Integer> path) {
        return nodeAt(path).children;
    }

    /** 节点没有精确位置时，回退到最近具有已知位置的真实祖先。 */
    public SourceLocation locationOf(List<Integer> path) {
        nodeAt(path);
        Node current = root;
        SourceLocation location = current.sourceLocation;
        for (Integer index : path) {
            current = current.children.get(index);
            if (current.sourceLocation.isKnown()) {
                location = current.sourceLocation;
            }
        }
        return location;
    }

    /** 用确定规则呈现中立记录；括号、引号和转义是呈现，不新增原文件字符坐标。 */
    public String plainText() {
        return root.plainText();
    }

    private static boolean findPath(Node current, Node target, List<Integer> path) {
        if (current == target) {
            return true;
        }
        for (int index = 0; index < current.children.size(); index++) {
            path.add(index);
            if (findPath(current.children.get(index), target, path)) {
                return true;
            }
            path.removeLast();
        }
        return false;
    }

    private static void validateTree(Node node, Set<Node> members) {
        require(members.add(node));
        for (Node child : node.children) {
            validateTree(child, members);
        }
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    /** 已解析的内容节点，不保存 JSONObject、Map 或第三方 SDK 节点。 */
    @Getter
    public static final class Node {
        public enum Kind {
            /** 对象/映射，直接子节点具有实际字段名。 */ OBJECT,
            /** 数组/序列，直接子节点按源顺序定位。 */ SEQUENCE,
            /** XML 元素，保留属性、子元素和文本的相对顺序。 */ ELEMENT,
            /** XML 属性，与同名子元素的语义不同。 */ ATTRIBUTE,
            /** XML 混合内容中的实际文本，不自动 trim。 */ TEXT,
            /** 有明确业务值类型的标量。 */ VALUE
        }

        public enum ValueType {
            /** 已解码的字符串。 */ STRING,
            /** 原始数字词法值，不转换为浮点数。 */ NUMBER,
            /** 已确认的 true/false。 */ BOOLEAN,
            /** 已确认的空值，不等同于字符串 "null"。 */ NULL
        }

        /** 实际节点种类，决定值和子节点的合法组合。 */
        private final Kind kind;
        /** 实际字段或元素名称；数组项和混合文本可以没有名称。 */
        private final String name;
        /** 标量值类型；容器没有此字段。 */
        private final ValueType valueType;
        /** 原始业务值，数字保留词法精度；NULL 和容器为 null。 */
        private final String value;
        /** 不可变、按源顺序排列的直接子节点。 */
        private final List<Node> children;
        /** 适配器实际证明的节点位置；未知不补造。 */
        private final SourceLocation sourceLocation;

        private Node(Kind kind, String name, ValueType valueType, String value,
                     List<Node> children, SourceLocation location) {
            require(kind != null && children != null && children.stream().noneMatch(child -> child == null));
            require(name == null || !name.isBlank());
            boolean container = kind == Kind.OBJECT || kind == Kind.SEQUENCE || kind == Kind.ELEMENT;
            require(container ? valueType == null && value == null : children.isEmpty() && valueType != null);
            require(kind != Kind.ELEMENT && kind != Kind.ATTRIBUTE || name != null);
            require(kind != Kind.TEXT || name == null);
            if (!container) {
                validateValue(valueType, value);
                require(kind == Kind.VALUE || valueType == ValueType.STRING);
            }
            if (kind == Kind.OBJECT) {
                require(children.stream().allMatch(child -> child.name != null
                        && child.kind != Kind.ATTRIBUTE && child.kind != Kind.TEXT));
            } else if (kind == Kind.SEQUENCE) {
                require(children.stream().allMatch(child -> child.kind != Kind.ATTRIBUTE && child.kind != Kind.TEXT));
            } else if (kind == Kind.ELEMENT) {
                require(children.stream().allMatch(child -> child.kind == Kind.ATTRIBUTE
                        || child.kind == Kind.TEXT || child.kind == Kind.ELEMENT));
                require(children.stream().filter(child -> child.kind == Kind.ATTRIBUTE).map(Node::getName)
                        .distinct().count() == children.stream().filter(child -> child.kind == Kind.ATTRIBUTE).count());
            }
            this.kind = kind;
            this.name = name;
            this.valueType = valueType;
            this.value = value;
            this.children = List.copyOf(children);
            this.sourceLocation = location == null ? SourceLocation.unknown() : location;
        }

        public static Node object(String name, List<Node> fields, SourceLocation location) {
            return new Node(Kind.OBJECT, name, null, null, fields, location);
        }

        public static Node sequence(String name, List<Node> items, SourceLocation location) {
            return new Node(Kind.SEQUENCE, name, null, null, items, location);
        }

        public static Node element(String name, List<Node> content, SourceLocation location) {
            return new Node(Kind.ELEMENT, name, null, null, content, location);
        }

        public static Node attribute(String name, String value, SourceLocation location) {
            return new Node(Kind.ATTRIBUTE, name, ValueType.STRING, value, List.of(), location);
        }

        public static Node text(String value, SourceLocation location) {
            return new Node(Kind.TEXT, null, ValueType.STRING, value, List.of(), location);
        }

        public static Node value(String name, ValueType type, String value, SourceLocation location) {
            return new Node(Kind.VALUE, name, type, value, List.of(), location);
        }

        /** 呈现实际节点内容，XML 不降成 JSON；名称也是事实，但呈现格式不视为源字符。 */
        public String plainText() {
            if (kind == Kind.ELEMENT) {
                StringBuilder result = new StringBuilder("<").append(name);
                children.stream().filter(child -> child.kind == Kind.ATTRIBUTE)
                        .forEach(child -> result.append(' ').append(child.plainText()));
                result.append('>');
                children.stream().filter(child -> child.kind != Kind.ATTRIBUTE)
                        .forEach(child -> result.append(child.plainText()));
                return result.append("</").append(name).append('>').toString();
            }
            if (kind == Kind.ATTRIBUTE) {
                return name + "=\"" + xml(value).replace("\"", "&quot;") + "\"";
            }
            if (kind == Kind.TEXT) {
                return xml(value);
            }
            String body = switch (kind) {
                case OBJECT -> "{" + String.join(", ", children.stream().map(Node::plainText).toList()) + "}";
                case SEQUENCE -> "[" + String.join(", ", children.stream().map(Node::plainText).toList()) + "]";
                case VALUE -> valueType == ValueType.NULL ? "null"
                        : valueType == ValueType.STRING ? quote(value) : value;
                default -> throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
            };
            return name == null ? body : quote(name) + ": " + body;
        }

        private static void validateValue(ValueType type, String value) {
            require(type != null && (type == ValueType.NULL ? value == null : value != null));
            if (type == ValueType.BOOLEAN) {
                require("true".equals(value) || "false".equals(value));
            } else if (type == ValueType.NUMBER) {
                require(!value.isBlank() && value.equals(value.trim()));
                try {
                    new BigDecimal(value);
                } catch (NumberFormatException exception) {
                    throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
                }
            }
        }

        private static String quote(String text) {
            // 复用 common-util 已提供的 Fastjson2 字符串转义，不通过反射读写领域聚合。
            return JSON.toJSONString(text);
        }

        private static String xml(String text) {
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
        }
    }
}
