import React from 'react';
import {
  ActivityIndicator,
  Dimensions,
  Platform,
  Pressable,
  StyleProp,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius } from '../theme';

if (Platform.OS === 'web' && typeof document !== 'undefined' && !document.getElementById('app-global-css')) {
  const style = document.createElement('style');
  style.id = 'app-global-css';
  style.textContent =
    '*::-webkit-scrollbar{display:none}*{scrollbar-width:none;-ms-overflow-style:none}body{background:#E5E7EB}';
  document.head.appendChild(style);
}

export function Screen({ children }: { children: React.ReactNode }) {
  if (Platform.OS === 'web') {
    return (
      <View style={styles.webShell}>
        <View style={styles.webPhone}>{children}</View>
      </View>
    );
  }
  return <View style={{ flex: 1, backgroundColor: colors.bg }}>{children}</View>;
}

export function Header({ title, onBack, dark }: { title: string; onBack?: () => void; dark?: boolean }) {
  const fg = dark ? '#F9FAFB' : colors.ink;
  return (
    <View style={[styles.header, dark && { backgroundColor: colors.navy }]}>
      {onBack ? (
        <Pressable onPress={onBack} hitSlop={12} style={[styles.headerBtn, dark && { backgroundColor: '#1F2937' }]}>
          <Ionicons name="chevron-back" size={22} color={fg} />
        </Pressable>
      ) : (
        <View style={{ width: 40 }} />
      )}
      <Text style={[styles.headerTitle, dark && { color: fg }]} numberOfLines={1}>
        {title}
      </Text>
      <View style={{ width: 40 }} />
    </View>
  );
}

export function Card({ children, style }: { children: React.ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

type ButtonProps = {
  label: string;
  onPress?: () => void;
  variant?: 'primary' | 'success' | 'outline' | 'ghost';
  icon?: keyof typeof Ionicons.glyphMap;
  disabled?: boolean;
  loading?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function Button({ label, onPress, variant = 'primary', icon, disabled, loading, style }: ButtonProps) {
  const bg =
    variant === 'primary' ? colors.primary : variant === 'success' ? colors.success : 'transparent';
  const fg = variant === 'outline' || variant === 'ghost' ? colors.primary : '#FFFFFF';
  return (
    <Pressable
      onPress={disabled || loading ? undefined : onPress}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: bg, opacity: disabled ? 0.45 : pressed ? 0.85 : 1 },
        variant === 'outline' && { borderWidth: 1.5, borderColor: colors.primary },
        variant === 'ghost' && { backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border },
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={fg} size="small" />
      ) : (
        <>
          {icon ? <Ionicons name={icon} size={18} color={fg} style={{ marginRight: 6 }} /> : null}
          <Text style={[styles.buttonText, { color: fg }]}>{label}</Text>
        </>
      )}
    </Pressable>
  );
}

export function Chip({
  label,
  tone = 'neutral',
  icon,
  active,
  onPress,
}: {
  label: string;
  tone?: 'neutral' | 'primary' | 'success' | 'warning' | 'danger';
  icon?: keyof typeof Ionicons.glyphMap;
  active?: boolean;
  onPress?: () => void;
}) {
  const tones = {
    neutral: { c: colors.textMuted, b: colors.border },
    primary: { c: colors.primary, b: colors.primary },
    success: { c: colors.successDark, b: colors.success },
    warning: { c: '#B45309', b: colors.warning },
    danger: { c: colors.danger, b: colors.danger },
  }[tone];
  const filled = active || tone === 'primary';
  return (
    <Pressable
      onPress={onPress}
      style={[
        styles.chip,
        {
          borderColor: tones.b,
          backgroundColor: filled && active ? tones.b : colors.surface,
        },
      ]}
    >
      {icon ? (
        <Ionicons name={icon} size={13} color={filled && active ? '#FFF' : tones.c} style={{ marginRight: 5 }} />
      ) : null}
      <Text style={[styles.chipText, { color: filled && active ? '#FFF' : tones.c }]}>{label}</Text>
    </Pressable>
  );
}

export function StatusPill({ status }: { status: 'new' | 'pending' | 'approved' | 'anomaly' }) {
  const map = {
    new: { label: 'Новая', icon: 'ellipse' as const, c: colors.primary },
    pending: { label: 'На согласовании', icon: 'time-outline' as const, c: colors.warning },
    approved: { label: 'Согласовано', icon: 'checkmark-circle' as const, c: colors.success },
    anomaly: { label: 'Аномалия', icon: 'warning' as const, c: colors.danger },
  }[status];
  return (
    <View style={[styles.pill, { borderColor: map.c }]}>
      <Ionicons name={map.icon} size={13} color={map.c} style={{ marginRight: 5 }} />
      <Text style={[styles.pillText, { color: map.c }]}>{map.label}</Text>
    </View>
  );
}

export function AiBadge({ text }: { text: string }) {
  return (
    <View style={styles.aiBadge}>
      <Ionicons name="sparkles" size={13} color="#FFF" style={{ marginRight: 5 }} />
      <Text style={styles.aiBadgeText}>{text}</Text>
    </View>
  );
}

const webPhoneHeight = Math.min(Dimensions.get('window').height - 48, 840);

const styles = StyleSheet.create({
  webShell: {
    flex: 1,
    backgroundColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 24,
  },
  webPhone: {
    width: 400,
    height: webPhoneHeight,
    backgroundColor: colors.bg,
    borderRadius: 36,
    overflow: 'hidden',
    borderWidth: 8,
    borderColor: '#111827',
    shadowColor: '#000',
    shadowOpacity: 0.25,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 10 },
    elevation: 12,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingTop: Platform.select({ web: 16, default: 52 }),
    paddingBottom: 10,
    backgroundColor: colors.bg,
  },
  headerBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surface,
  },
  headerTitle: { fontSize: 18, fontWeight: '700', color: colors.text, flex: 1, textAlign: 'center' },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.card,
    padding: 16,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.input,
    paddingVertical: 14,
    paddingHorizontal: 16,
  },
  buttonText: { fontSize: 15, fontWeight: '700' },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.2,
    borderRadius: radius.pill,
    paddingVertical: 7,
    paddingHorizontal: 12,
  },
  chipText: { fontSize: 13, fontWeight: '600' },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.2,
    borderRadius: radius.pill,
    paddingVertical: 3,
    paddingHorizontal: 9,
    alignSelf: 'flex-start',
  },
  pillText: { fontSize: 12, fontWeight: '600' },
  aiBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.success,
    borderRadius: radius.pill,
    paddingVertical: 6,
    paddingHorizontal: 10,
    alignSelf: 'flex-start',
  },
  aiBadgeText: { color: '#FFF', fontSize: 12, fontWeight: '700' },
});
