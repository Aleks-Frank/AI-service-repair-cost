import React, { useMemo, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Image } from 'expo-image';
import { Ionicons } from '@expo/vector-icons';
import { AiBadge, Card, Chip, Screen, StatusPill } from '../components/ui';
import { colors } from '../theme';
import { formatMoney, useStore } from '../store';
import type { ScreenProps } from '../navigation/types';

type Filter = 'all' | 'pending' | 'approved' | 'anomaly';

export function Dashboard({ navigation }: ScreenProps<'Dashboard'>) {
  const { requests } = useStore();
  const [filter, setFilter] = useState<Filter>('all');

  const filtered = useMemo(
    () => (filter === 'all' ? requests : requests.filter((r) => r.status === filter)),
    [requests, filter]
  );

  const avgAccuracy = useMemo(() => {
    const withAcc = requests.filter((r) => r.accuracy);
    return Math.round(withAcc.reduce((s, r) => s + (r.accuracy || 0), 0) / Math.max(1, withAcc.length));
  }, [requests]);

  return (
    <Screen>
      <View style={styles.topBar}>
        <Text style={styles.title}>Заявки</Text>
        <View style={{ flexDirection: 'row', gap: 8 }}>
          <Pressable style={styles.iconBtn}>
            <Ionicons name="search" size={20} color={colors.ink} />
          </Pressable>
          <Pressable style={styles.iconBtn}>
            <Ionicons name="notifications-outline" size={20} color={colors.ink} />
            <View style={styles.bellDot} />
          </Pressable>
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.kpiRow}>
          <Card style={styles.kpi}>
            <Text style={styles.kpiLabel}>Время до сметы</Text>
            <Text style={styles.kpiOld}>30–90 мин</Text>
            <Text style={[styles.kpiValue, { color: colors.success }]}>→ 8 сек</Text>
          </Card>
          <Card style={styles.kpi}>
            <Text style={styles.kpiLabel}>Новых заявок</Text>
            <Text style={[styles.kpiValue, { color: colors.primary, marginTop: 10 }]}>
              {requests.filter((r) => r.status === 'new' || r.status === 'pending').length}
            </Text>
          </Card>
          <Card style={styles.kpi}>
            <Text style={styles.kpiLabel}>Средняя точность</Text>
            <Text style={[styles.kpiValue, { color: colors.success, marginTop: 10 }]}>{avgAccuracy}%</Text>
          </Card>
        </View>

        <View style={styles.filters}>
          <Chip label="Все" tone="primary" active={filter === 'all'} onPress={() => setFilter('all')} />
          <Chip label="На согласовании" tone="warning" active={filter === 'pending'} onPress={() => setFilter('pending')} />
          <Chip label="Согласовано" tone="success" active={filter === 'approved'} onPress={() => setFilter('approved')} />
          <Chip label="Аномалии" tone="danger" icon="warning" active={filter === 'anomaly'} onPress={() => setFilter('anomaly')} />
        </View>

        {filtered.map((r) => (
          <Card key={r.id} style={styles.reqCard}>
            <Image source={r.photo} style={styles.thumb} contentFit="cover" transition={150} />
            <View style={{ flex: 1 }}>
              <Text style={styles.reqTitle}>{r.title}</Text>
              <Text style={styles.reqAddr}>{r.address}</Text>
              <View style={{ marginTop: 6 }}>
                <StatusPill status={r.status} />
              </View>
              {r.anomalyNote ? (
                <View style={styles.anomalyLine}>
                  <Ionicons name="alert-circle" size={13} color={colors.warning} />
                  <Text style={styles.anomalyText}>Аномалия: {r.anomalyNote}</Text>
                </View>
              ) : null}
              <Text style={styles.reqDate}>{r.createdAt}</Text>
            </View>
            <View style={styles.reqRight}>
              <Text style={styles.reqTotal}>≈ {formatMoney(r.total)} ₽</Text>
              <AiBadge text="AI" />
            </View>
          </Card>
        ))}
        <View style={{ height: 90 }} />
      </ScrollView>

      <Pressable style={styles.fab} onPress={() => navigation.navigate('NewRequest')}>
        <Ionicons name="camera" size={26} color="#FFF" />
        <View style={styles.fabPlus}>
          <Ionicons name="add" size={13} color="#FFF" />
        </View>
      </Pressable>
    </Screen>
  );
}

const styles = StyleSheet.create({
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'web' ? 20 : 56,
    paddingBottom: 12,
  },
  title: { fontSize: 26, fontWeight: '800', color: colors.text },
  iconBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
  bellDot: { position: 'absolute', top: 9, right: 11, width: 8, height: 8, borderRadius: 4, backgroundColor: colors.warning },
  scroll: { paddingHorizontal: 16, flexGrow: 1 },
  kpiRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  kpi: { flex: 1, padding: 12 },
  kpiLabel: { fontSize: 10.5, color: colors.textMuted, marginBottom: 2 },
  kpiOld: { fontSize: 11.5, color: colors.textMuted, textDecorationLine: 'line-through' },
  kpiValue: { fontSize: 15, fontWeight: '800', color: colors.text },
  filters: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 12 },
  reqCard: { flexDirection: 'row', padding: 12, marginBottom: 10, gap: 12 },
  thumb: { width: 64, height: 64, borderRadius: 12, backgroundColor: colors.border },
  reqTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  reqAddr: { fontSize: 12, color: colors.textMuted, marginTop: 2 },
  anomalyLine: { flexDirection: 'row', gap: 4, alignItems: 'center', marginTop: 6 },
  anomalyText: { fontSize: 11.5, color: '#B45309', fontWeight: '600' },
  reqDate: { fontSize: 11, color: colors.textMuted, marginTop: 6 },
  reqRight: { alignItems: 'flex-end', justifyContent: 'space-between', gap: 8 },
  reqTotal: { fontSize: 15, fontWeight: '800', color: colors.text },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 28,
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 6,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
  },
  fabPlus: {
    position: 'absolute',
    top: -2,
    right: -2,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: colors.success,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
